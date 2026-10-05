package io.kestra.plugin.ai.embeddings;

import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.Map;

import javax.net.ssl.SSLContext;

import org.apache.http.Header;
import org.apache.http.HttpEntity;
import org.apache.http.HttpEntityEnclosingRequest;
import org.apache.http.HttpHost;
import org.apache.http.HttpRequest;
import org.apache.http.HttpRequestInterceptor;
import org.apache.http.entity.AbstractHttpEntity;
import org.apache.http.auth.AuthScope;
import org.apache.http.auth.UsernamePasswordCredentials;
import org.apache.http.client.CredentialsProvider;
import org.apache.http.conn.ssl.NoopHostnameVerifier;
import org.apache.http.conn.ssl.TrustStrategy;
import org.apache.http.impl.client.BasicCredentialsProvider;
import org.apache.http.impl.nio.client.HttpAsyncClientBuilder;
import org.apache.http.message.BasicHeader;
import org.apache.http.ssl.SSLContextBuilder;
import org.elasticsearch.client.Request;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestClientBuilder;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;

import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.instrumentation.NoopInstrumentation;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.SneakyThrows;
import lombok.experimental.SuperBuilder;

// it needs Elasticsearch 8.15 min
@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Store embeddings in Elasticsearch",
    description = "Targets an Elasticsearch 8.15+ cluster using the provided hosts/index; when `drop=true` the index is deleted. Supports basic auth, custom headers, path prefix, and trust-all TLS for self-signed certs."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Ingest documents into an Elasticsearch embedding store (requires Elasticsearch 8.15+)",
            code = """
                id: document_ingestion
                namespace: company.ai

                tasks:
                  - id: ingest
                    type: io.kestra.plugin.ai.rag.IngestDocument
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-001
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.Elasticsearch
                      connection:
                        hosts:
                          - http://localhost:9200
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        ),
    },
    aliases = "io.kestra.plugin.langchain4j.embeddings.Elasticsearch"
)
public class Elasticsearch extends EmbeddingStoreProvider {
    @JsonIgnore
    private transient RestClient restClient;

    @NotNull
    @Schema(
        title = "Connection",
        description = "Elasticsearch connection settings: hosts, authentication, and TLS options. No default: this property is required.",
        example = "{hosts: [\"http://localhost:9200\"]}"
    )
    private ElasticsearchConnection connection;

    @NotNull
    @Schema(
        title = "Index name",
        description = "Elasticsearch index that stores the embeddings. No default: this property is required.",
        example = "embeddings"
    )
    @PluginProperty(group = "main")
    private Property<String> indexName;

    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) throws IOException, IllegalVariableEvaluationException {
        restClient = connection.client(runContext).restClient();

        if (drop) {
            restClient.performRequest(new Request("DELETE", runContext.render(indexName).as(String.class).orElseThrow()));
        }

        return dev.langchain4j.store.embedding.elasticsearch.ElasticsearchEmbeddingStore.builder()
            .restClient(restClient)
            .indexName(runContext.render(indexName).as(String.class).orElseThrow())
            .build();
    }

    @Override
    public Map<String, Object> outputs(RunContext runContext) throws IOException {
        if (restClient != null) {
            restClient.close();
        }

        return null;
    }

    // Copy of o.kestra.plugin.elasticsearch.ElasticsearchConnection
    @Builder
    @Getter
    public static class ElasticsearchConnection {
        private static final ObjectMapper MAPPER = JacksonMapper.ofJson(false);
        private static final Logger log = LoggerFactory.getLogger(ElasticsearchConnection.class);

        @Schema(
            title = "Elasticsearch HTTP servers",
            description = "URLs of the Elasticsearch nodes to connect to, each including scheme, host and port. No default: this property is required and must not be empty.",
            example = "[\"https://elasticsearch.internal:9200\"]"
        )
        @PluginProperty(dynamic = true, group = "main")
        @NotNull
        @NotEmpty
        private List<String> hosts;

        @Schema(
            title = "Basic authorization",
            description = "Username and password used for HTTP basic authentication. Not set by default (anonymous access).",
            example = "{username: \"elastic\", password: \"{{ secret('ES_PASSWORD') }}\"}"
        )
        @PluginProperty(group = "advanced")
        private BasicAuth basicAuth;

        @Schema(
            title = "HTTP headers sent with every request",
            description = "Extra HTTP headers added to each request, each written as a `key: value` string. Not set by default.",
            example = "[\"Authorization: Token XYZ\"]"
        )
        @PluginProperty(group = "advanced")
        private Property<List<String>> headers;

        @Schema(
            title = "Path prefix for all HTTP requests",
            description = "Prefix prepended to every request path, so `/my/path` turns each call into `/my/path/` + endpoint. Use it only when Elasticsearch sits behind a proxy that serves it under a base path. Not set by default.",
            example = "/my/path"
        )
        @PluginProperty(group = "advanced")
        private Property<String> pathPrefix;

        @Schema(
            title = "Strict deprecation mode",
            description = "If `true`, responses carrying deprecation warnings are treated as failures. Defaults to `false`.",
            example = "false"
        )
        @PluginProperty(group = "advanced")
        private Property<Boolean> strictDeprecationMode;

        @Schema(
            title = "Trust all SSL CA certificates",
            description = "If `true`, accept any TLS certificate presented by the server, which is sometimes needed for self-signed certificates. Defaults to `false`. WARNING: enabling this disables both certificate chain validation and hostname verification, exposing connections to man-in-the-middle attacks. Prefer supplying a custom CA certificate instead, and use this only in trusted, controlled environments.",
            example = "false",
            deprecated = true
        )
        @PluginProperty(group = "advanced")
        @Deprecated
        private Property<Boolean> trustAllSsl;

        @Schema(
            title = "Target Elasticsearch server major version",
            description = "Major version advertised in the `compatible-with` media-type headers (`Accept` and `Content-Type`). The bundled `elasticsearch-java` 9.x client would otherwise negotiate `compatible-with=9`, which Elasticsearch 8 rejects. Use `8` for an Elasticsearch 8 cluster or `9` for Elasticsearch 9. Defaults to `8`.",
            example = "8"
        )
        @PluginProperty(group = "advanced")
        @Builder.Default
        private Property<Integer> targetServerVersion = Property.ofValue(8);

        @SuperBuilder
        @NoArgsConstructor
        @Getter
        public static class BasicAuth {
            @Schema(
                title = "Basic authorization username",
                description = "User authenticating against Elasticsearch. No default: this property is required inside `basicAuth`.",
                example = "elastic"
            )
            @PluginProperty(group = "connection")
            private Property<String> username;

            @Schema(
                title = "Basic authorization password",
                description = "Password of the Elasticsearch user. Store it as a Kestra secret rather than inline. No default: this property is required inside `basicAuth`.",
                example = "{{ secret('ELASTICSEARCH_PASSWORD') }}"
            )
            @PluginProperty(secret = true, group = "connection")
            private Property<String> password;
        }

        RestClientTransport client(RunContext runContext) throws IllegalVariableEvaluationException {
            RestClientBuilder builder = RestClient
                .builder(this.httpHosts(runContext))
                .setHttpClientConfigCallback(httpClientBuilder ->
                {
                    httpClientBuilder = this.httpAsyncClientBuilder(runContext);
                    return httpClientBuilder;
                });

            if (this.getHeaders() != null) {
                builder.setDefaultHeaders(this.defaultHeaders(runContext));
            }

            if (runContext.render(this.pathPrefix).as(String.class).isPresent()) {
                builder.setPathPrefix(runContext.render(this.pathPrefix).as(String.class).get());
            }

            if (runContext.render(this.strictDeprecationMode).as(Boolean.class).isPresent()) {
                builder.setStrictDeprecationMode(runContext.render(this.strictDeprecationMode).as(Boolean.class).get());
            }

            return new RestClientTransport(
                builder.build(), new JacksonJsonpMapper(MAPPER), null,
                NoopInstrumentation.INSTANCE
            );
        }

        @SneakyThrows
        private HttpAsyncClientBuilder httpAsyncClientBuilder(RunContext runContext) {
            HttpAsyncClientBuilder builder = HttpAsyncClientBuilder.create();

            builder.setUserAgent("Kestra/" + runContext.version());

            if (basicAuth != null) {
                final CredentialsProvider basicCredential = new BasicCredentialsProvider();
                basicCredential.setCredentials(
                    AuthScope.ANY,
                    new UsernamePasswordCredentials(
                        runContext.render(this.basicAuth.username).as(String.class).orElseThrow(),
                        runContext.render(this.basicAuth.password).as(String.class).orElseThrow()
                    )
                );

                builder.setDefaultCredentialsProvider(basicCredential);
            }

            if (runContext.render(this.trustAllSsl).as(Boolean.class).orElse(false)) {
                log.warn(
                    "trustAllSsl=true: TLS certificate chain validation and hostname verification are DISABLED. "
                    + "This exposes the connection to man-in-the-middle attacks. "
                    + "Use only in controlled environments with self-signed certificates. "
                    + "This option is deprecated and will be removed in a future release."
                );
                SSLContextBuilder sslContextBuilder = new SSLContextBuilder();
                sslContextBuilder.loadTrustMaterial(null, (TrustStrategy) (chain, authType) -> true);
                SSLContext sslContext = sslContextBuilder.build();

                builder.setSSLContext(sslContext);
                builder.setSSLHostnameVerifier(new NoopHostnameVerifier());
            }

            // elasticsearch-java 9.x hard-codes compatible-with=9 in media-type headers, which
            // Elasticsearch 8 clusters reject. Intercept every request and rewrite the version.
            // Content-Type is set on the HttpEntity (not the request headers), so we must rewrite
            // it there; Accept is a plain request header and is handled by rewriteCompatibleWith.
            int targetVersion = runContext.render(this.targetServerVersion).as(Integer.class).orElse(8);
            builder.addInterceptorFirst((HttpRequestInterceptor) (request, context) -> {
                rewriteCompatibleWith(request, "Content-Type", targetVersion);
                rewriteCompatibleWith(request, "Accept", targetVersion);
                if (request instanceof HttpEntityEnclosingRequest entityRequest) {
                    HttpEntity entity = entityRequest.getEntity();
                    if (entity instanceof AbstractHttpEntity abstractEntity) {
                        Header ct = abstractEntity.getContentType();
                        if (ct != null && ct.getValue().contains("compatible-with=")) {
                            abstractEntity.setContentType(ct.getValue().replaceFirst("compatible-with=\\d+", "compatible-with=" + targetVersion));
                        }
                    }
                }
            });

            return builder;
        }

        private static void rewriteCompatibleWith(HttpRequest request, String headerName, int version) {
            Header header = request.getFirstHeader(headerName);
            if (header != null && header.getValue().contains("compatible-with=")) {
                request.removeHeaders(headerName);
                request.addHeader(headerName, header.getValue().replaceFirst("compatible-with=\\d+", "compatible-with=" + version));
            }
        }

        private HttpHost[] httpHosts(RunContext runContext) throws IllegalVariableEvaluationException {
            return runContext.render(this.hosts)
                .stream()
                .map(s ->
                {
                    URI uri = URI.create(s);
                    return new HttpHost(uri.getHost(), uri.getPort(), uri.getScheme());
                })
                .toArray(HttpHost[]::new);
        }

        private Header[] defaultHeaders(RunContext runContext) throws IllegalVariableEvaluationException {
            return runContext.render(this.headers).asList(String.class)
                .stream()
                .map(header ->
                {
                    String[] nameAndValue = header.split(":");
                    return new BasicHeader(nameAndValue[0], nameAndValue[1]);
                })
                .toArray(Header[]::new);
        }
    }
}
