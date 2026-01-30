package io.kestra.plugin.ai.embeddings;

import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.instrumentation.NoopInstrumentation;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.SneakyThrows;
import lombok.experimental.SuperBuilder;
import org.apache.http.Header;
import org.apache.http.HttpHost;
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

import javax.net.ssl.SSLContext;
import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.Map;

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
                      modelName: gemini-embedding-exp-03-07
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
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
    private ElasticsearchConnection connection;

    @NotNull
    @Schema(title = "The name of the index to store embeddings")
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

        @Schema(
            title = "List of HTTP Elasticsearch servers",
            description = "Must be a URI like `https://example.com:9200` with scheme and port"
        )
        @PluginProperty(dynamic = true)
        @NotNull
        @NotEmpty
        private List<String> hosts;

        @Schema(title = "Basic authorization configuration")
        @PluginProperty
        private BasicAuth basicAuth;

        @Schema(
            title = "List of HTTP headers to be sent with every request",
            description = "Each item is a `key: value` string, e.g., `Authorization: Token XYZ`"
        )
        private Property<List<String>> headers;

        @Schema(
            title = "Path prefix for all HTTP requests",
            description = "If set to `/my/path`, each client request becomes `/my/path/` + endpoint. Useful when Elasticsearch is behind a proxy providing a base path; do not use otherwise."
        )
        private Property<String> pathPrefix;

        @Schema(
            title = "Treat responses with deprecation warnings as failures"
        )
        private Property<Boolean> strictDeprecationMode;

        @Schema(
            title = "Trust all SSL CA certificates",
            description = "Use this if the server uses a self-signed SSL certificate"
        )
        private Property<Boolean> trustAllSsl;

        @SuperBuilder
        @NoArgsConstructor
        @Getter
        public static class BasicAuth {
            @Schema(title = "Basic authorization username")
            private Property<String> username;

            @Schema(title = "Basic authorization password")
            private Property<String> password;
        }

        RestClientTransport client(RunContext runContext) throws IllegalVariableEvaluationException {
            RestClientBuilder builder = RestClient
                .builder(this.httpHosts(runContext))
                .setHttpClientConfigCallback(httpClientBuilder -> {
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

            return new RestClientTransport(builder.build(), new JacksonJsonpMapper(MAPPER), null,
                NoopInstrumentation.INSTANCE);
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
                SSLContextBuilder sslContextBuilder = new SSLContextBuilder();
                sslContextBuilder.loadTrustMaterial(null, (TrustStrategy) (chain, authType) -> true);
                SSLContext sslContext = sslContextBuilder.build();

                builder.setSSLContext(sslContext);
                builder.setSSLHostnameVerifier(new NoopHostnameVerifier());
            }

            return builder;
        }

        private HttpHost[] httpHosts(RunContext runContext) throws IllegalVariableEvaluationException {
            return runContext.render(this.hosts)
                .stream()
                .map(s -> {
                    URI uri = URI.create(s);
                    return new HttpHost(uri.getHost(), uri.getPort(), uri.getScheme());
                })
                .toArray(HttpHost[]::new);
        }

        private Header[] defaultHeaders(RunContext runContext) throws IllegalVariableEvaluationException {
            return runContext.render(this.headers).asList(String.class)
                .stream()
                .map(header -> {
                    String[] nameAndValue = header.split(":");
                    return new BasicHeader(nameAndValue[0], nameAndValue[1]);
                })
                .toArray(Header[]::new);
        }
    }
}
