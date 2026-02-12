package io.kestra.plugin.ai.embeddings;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.weaviate.WeaviateEmbeddingStore;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.io.IOException;
import java.util.List;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Store embeddings in Weaviate",
    description = "Connects to a Weaviate cluster (HTTP + optional gRPC) using the given host/scheme. Defaults: objectClass \"Default\", avoidDups true, consistency QUORUM. Provide API key when auth is enabled; `drop=true` clears the class contents."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Ingest documents into a Weaviate embedding store",
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
                      type: io.kestra.plugin.ai.embeddings.Weaviate
                      apiKey: "{{ kv('WEAVIATE_API_KEY') }}"   # omit for local/no-auth
                      scheme: https                                 # http | https
                      host: your-cluster-id.weaviate.network        # no protocol
                      # port: 443                                   # optional; usually omit
                    drop: true
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        )
    },
    aliases = "io.kestra.plugin.langchain4j.embeddings.Weaviate"
)
public class Weaviate extends EmbeddingStoreProvider {

    @Schema(
        title = "API key",
        description = "Weaviate API key. Omit for local deployments without auth."
    )
    @NotNull
    private Property<String> apiKey;

    @Schema(
        title = "Scheme",
        description = "Cluster scheme: \"https\" (recommended) or \"http\"."
    )
    private Property<String> scheme;

    @Schema(
        title = "Host",
        description = "Cluster host name without protocol, e.g., \"abc123.weaviate.network\"."
    )
    @NotNull
    private Property<String> host;

    @Schema(
        title = "Port",
        description = "Optional port (e.g., 443 for https, 80 for http). Leave unset to use provider defaults."
    )
    private Property<Integer> port;

    @Schema(
        title = "Object class",
        description = "Weaviate class to store objects in (must start with an uppercase letter). Defaults to \"Default\" if not set."
    )
    private Property<String> objectClass;

    @Schema(
        title = "Consistency level",
        description = "Write consistency: ONE, QUORUM (default), or ALL."
    )
    private Property<ConsistencyLevel> consistencyLevel;

    @Schema(
        title = "Avoid duplicates",
        description = "If true (default), a hash-based ID is derived from each text segment to prevent duplicates. If false, a random ID is used."
    )
    private Property<Boolean> avoidDups;

    @Schema(
        title = "Metadata field name",
        description = "Field used to store metadata. Defaults to \"_metadata\" if not set."
    )
    private Property<String> metadataFieldName;

    @Schema(
        title = "Metadata keys",
        description = "The list of metadata keys to store - if not provided, it will default to an empty list."
    )
    private Property<List<String>> metadataKeys;

    @Schema(
        title = "Use gRPC for batch inserts",
        description = "If true, use gRPC for batch inserts. HTTP remains required for search operations."
    )
    private Property<Boolean> useGrpcForInserts;

    @Schema(
        title = "Secure gRPC",
        description = "Whether the gRPC connection is secured (TLS)."
    )
    private Property<Boolean> securedGrpc;

    @Schema(
        title = "gRPC port",
        description = "Port for gRPC if enabled (e.g., 50051)."
    )
    private Property<Integer> grpcPort;

    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) throws IOException, IllegalVariableEvaluationException {

        // dimension is useless since the given embedding dimension will be used inside Qdrant

        var store = WeaviateEmbeddingStore.builder()
            .apiKey(runContext.render(apiKey).as(String.class).orElseThrow())
            .scheme(runContext.render(scheme).as(String.class).orElse("https"))
            .host(runContext.render(host).as(String.class).orElseThrow())
            .port(runContext.render(port).as(Integer.class).orElseThrow())
            .objectClass(runContext.render(objectClass).as(String.class).orElseThrow())
            .avoidDups(runContext.render(avoidDups).as(Boolean.class).orElse(true))
            .consistencyLevel(runContext.render(consistencyLevel).as(ConsistencyLevel.class).orElse(ConsistencyLevel.QUORUM).name())
            .metadataFieldName(runContext.render(metadataFieldName).as(String.class).orElse(null))
            .metadataKeys(runContext.render(metadataKeys).asList(String.class))
            .useGrpcForInserts(runContext.render(useGrpcForInserts).as(Boolean.class).orElse(false))
            .securedGrpc(runContext.render(securedGrpc).as(Boolean.class).orElse(true))
            .grpcPort(runContext.render(grpcPort).as(Integer.class).orElse(null))
            .build();

        if (drop) {
            store.removeAll();
        }

        return store;
    }

    enum ConsistencyLevel {
        ONE,
        QUORUM,
        ALL,
    }
}
