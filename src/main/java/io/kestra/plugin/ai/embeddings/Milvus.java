package io.kestra.plugin.ai.embeddings;

import java.io.IOException;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.milvus.MilvusEmbeddingStore;
import io.milvus.common.clientenum.ConsistencyLevelEnum;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import io.kestra.core.models.annotations.PluginProperty;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Store embeddings in Milvus",
    description = "Connects via URI or host/port with token-based auth; creates the target collection if missing. Metric/index/consistency options map to Milvus settings; `drop=true` clears the collection. Use defaults for host=localhost, port=19530, secure gRPC unless overridden."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Ingest documents into a Milvus embedding store",
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
                      type: io.kestra.plugin.ai.embeddings.Milvus
                      # Use either `uri` or `host`/`port`:
                      # For gRPC (typical): milvus://localhost:19530
                      # For HTTP: http://localhost:9091
                      uri: "http://localhost:19200"
                      token: "{{ secret('MILVUS_TOKEN') }}"  # omit if auth is disabled
                      collectionName: embeddings
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        )
    },
    aliases = "io.kestra.plugin.langchain4j.embeddings.Milvus"
)
public class Milvus extends EmbeddingStoreProvider {

    @Schema(
        title = "Token",
        description = "Milvus auth token. Required if authentication is enabled; omit for local deployments without auth."
    )
    @NotNull
    @PluginProperty(secret = true, group = "main")
    private Property<String> token;

    @Schema(
        title = "URI",
        description = """
            Connection URI. Use either `uri` OR `host`/`port` (not both).
            Examples:
            - gRPC (typical): "milvus://host:19530"
            - HTTP: "http://host:9091"
            """
    )
    @PluginProperty(group = "advanced")
    private Property<String> uri;

    @Schema(
        title = "Host",
        description = "Milvus host name (used when `uri` is not set). Default: \"localhost\"."
    )
    @PluginProperty(group = "connection")
    private Property<String> host;

    @Schema(
        title = "Port",
        description = "Milvus port (used when `uri` is not set). Typical: 19530 (gRPC) or 9091 (HTTP). Default: 19530."
    )
    @PluginProperty(group = "connection")
    private Property<Integer> port;

    @Schema(
        title = "Username",
        description = "Required when authentication/TLS is enabled. See https://milvus.io/docs/authenticate.md"
    )
    @PluginProperty(secret = true, group = "connection")
    private Property<String> username;

    @Schema(
        title = "Password",
        description = "Required when authentication/TLS is enabled. See https://milvus.io/docs/authenticate.md"
    )
    @PluginProperty(secret = true, group = "connection")
    private Property<String> password;

    @Schema(
        title = "Collection name",
        description = "Target collection. Created automatically if it does not exist. Default: \"default\"."
    )
    @PluginProperty(group = "advanced")
    private Property<String> collectionName;

    @Schema(
        title = "Consistency level",
        description = "Read/write consistency level. Common values include STRONG, BOUNDED, or EVENTUALLY (depends on client/version)."
    )
    @PluginProperty(group = "advanced")
    private Property<String> consistencyLevel;

    @Schema(
        title = "Index type",
        description = "Vector index type (e.g., IVF_FLAT, IVF_SQ8, HNSW). Depends on Milvus deployment and dataset."
    )
    @PluginProperty(group = "advanced")
    private Property<String> indexType;

    @Schema(
        title = "Metric type",
        description = "Similarity metric (e.g., L2, IP, COSINE). Should match the embedding provider’s expected metric."
    )
    @PluginProperty(group = "advanced")
    private Property<String> metricType;

    @Schema(
        title = "Retrieve embeddings on search",
        description = "If true, return stored embeddings along with matches. Default: false."
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> retrieveEmbeddingsOnSearch;

    @Schema(
        title = "Database name",
        description = "Logical database to use. If not provided, the default database is used."
    )
    @PluginProperty(group = "advanced")
    private Property<String> databaseName;

    @Schema(
        title = "Auto flush on insert",
        description = "If true, flush after insert operations. Setting it to false can improve throughput."
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> autoFlushOnInsert;

    @Schema(
        title = "Auto flush on delete",
        description = "If true, flush after delete operations."
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> autoFlushOnDelete;

    @Schema(
        title = "ID field name",
        description = "Field name for document IDs. Default depends on collection schema."
    )
    @PluginProperty(group = "advanced")
    private Property<String> idFieldName;

    @Schema(
        title = "Text field name",
        description = "Field name for original text. Default depends on collection schema."
    )
    @PluginProperty(group = "advanced")
    private Property<String> textFieldName;

    @Schema(
        title = "Metadata field name",
        description = "Field name for metadata. Default depends on collection schema."
    )
    @PluginProperty(group = "advanced")
    private Property<String> metadataFieldName;

    @Schema(
        title = "Vector field name",
        description = "Field name for the embedding vector. Must match the index definition and embedding dimensionality."
    )
    @PluginProperty(group = "advanced")
    private Property<String> vectorFieldName;

    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) throws IOException, IllegalVariableEvaluationException {
        var store = MilvusEmbeddingStore.builder()
            .token(runContext.render(token).as(String.class).orElseThrow())
            .uri(runContext.render(uri).as(String.class).orElse(null))
            .host(runContext.render(host).as(String.class).orElse(null))
            .port(runContext.render(port).as(Integer.class).orElse(null))
            .username(runContext.render(username).as(String.class).orElse(null))
            .password(runContext.render(password).as(String.class).orElse(null))
            .collectionName(runContext.render(collectionName).as(String.class).orElse(null))
            .consistencyLevel(ConsistencyLevelEnum.valueOf(runContext.render(consistencyLevel).as(String.class).orElse("EVENTUALLY")))
            .indexType(IndexType.valueOf(runContext.render(indexType).as(String.class).orElse("FLAT")))
            .metricType(MetricType.valueOf(runContext.render(metricType).as(String.class).orElse("COSINE")))
            .retrieveEmbeddingsOnSearch(runContext.render(retrieveEmbeddingsOnSearch).as(Boolean.class).orElse(false))
            .databaseName(runContext.render(databaseName).as(String.class).orElse(null))
            .autoFlushOnInsert(runContext.render(autoFlushOnInsert).as(Boolean.class).orElse(false))
            .idFieldName(runContext.render(idFieldName).as(String.class).orElse(null))
            .textFieldName(runContext.render(textFieldName).as(String.class).orElse(null))
            .metadataFieldName(runContext.render(metadataFieldName).as(String.class).orElse(null))
            .vectorFieldName(runContext.render(vectorFieldName).as(String.class).orElse(null))
            .dimension(dimension)
            .build();

        if (drop) {
            store.removeAll();
        }

        return store;
    }
}
