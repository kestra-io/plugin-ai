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
                      uri: "http://localhost:9091"
                      token: "{{ secret('MILVUS_TOKEN') }}"
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
        description = "Milvus authentication token. Store it as a Kestra secret rather than inline. No default: this property is required.",
        example = "{{ secret('MILVUS_TOKEN') }}"
    )
    @NotNull
    @PluginProperty(secret = true, group = "main")
    private Property<String> token;

    @Schema(
        title = "URI",
        description = "Full connection URI of the Milvus server, such as `milvus://host:19530` for gRPC or `http://host:9091` for HTTP. Set either this property or `host`/`port`, not both. Not set by default.",
        example = "milvus://localhost:19530"
    )
    @PluginProperty(group = "advanced")
    private Property<String> uri;

    @Schema(
        title = "Host",
        description = "Hostname of the Milvus server, used when `uri` is not set. Not set by default, in which case the Milvus client's own default applies.",
        example = "localhost"
    )
    @PluginProperty(group = "connection")
    private Property<String> host;

    @Schema(
        title = "Port",
        description = "Port of the Milvus server, used when `uri` is not set. Typically `19530` for gRPC or `9091` for HTTP. Not set by default, in which case the Milvus client's own default applies.",
        example = "19530"
    )
    @PluginProperty(group = "connection")
    private Property<Integer> port;

    @Schema(
        title = "Username",
        description = "User authenticating against Milvus. Required only when authentication or TLS is enabled; see https://milvus.io/docs/authenticate.md. Not set by default.",
        example = "root"
    )
    @PluginProperty(group = "connection")
    private Property<String> username;

    @Schema(
        title = "Password",
        description = "Password of the Milvus user. Required only when authentication or TLS is enabled. Store it as a Kestra secret rather than inline. Not set by default.",
        example = "{{ secret('MILVUS_PASSWORD') }}"
    )
    @PluginProperty(secret = true, group = "connection")
    private Property<String> password;

    @Schema(
        title = "Collection name",
        description = "Collection that stores the embeddings. Not set by default, in which case the Milvus client's own default collection name applies.",
        example = "my-documents"
    )
    @PluginProperty(group = "advanced")
    private Property<String> collectionName;

    @Schema(
        title = "Consistency level",
        description = "Read/write consistency level applied to the collection: `STRONG`, `BOUNDED`, `SESSION` or `EVENTUALLY`. Defaults to `EVENTUALLY`.",
        example = "EVENTUALLY"
    )
    @PluginProperty(group = "advanced")
    private Property<String> consistencyLevel;

    @Schema(
        title = "Index type",
        description = "Vector index built on the collection, such as `FLAT`, `IVF_FLAT`, `IVF_SQ8`, `HNSW`, `DISKANN` or `AUTOINDEX`. The best choice depends on the deployment and dataset size. Defaults to `FLAT`.",
        example = "FLAT"
    )
    @PluginProperty(group = "advanced")
    private Property<String> indexType;

    @Schema(
        title = "Metric type",
        description = "Similarity metric used to compare vectors: `L2`, `IP`, `COSINE`, `HAMMING` or `JACCARD`. It should match the metric the embedding model was trained for. Defaults to `COSINE`.",
        example = "COSINE"
    )
    @PluginProperty(group = "advanced")
    private Property<String> metricType;

    @Schema(
        title = "Retrieve embeddings on search",
        description = "If `true`, search results also carry the stored embedding vectors. Defaults to `false`.",
        example = "false"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> retrieveEmbeddingsOnSearch;

    @Schema(
        title = "Database name",
        description = "Logical Milvus database holding the collection. Not set by default, in which case the server's default database is used.",
        example = "default"
    )
    @PluginProperty(group = "advanced")
    private Property<String> databaseName;

    @Schema(
        title = "Auto flush on insert",
        description = "If `true`, flush the collection after every insert so new vectors are immediately searchable. Setting it to `false` improves ingestion throughput. Defaults to `false`.",
        example = "false"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> autoFlushOnInsert;

    @Schema(
        title = "Auto flush on delete",
        description = "Intended to flush the collection after every delete. Note: this property currently has no effect, as it is not passed to the Milvus client when the store is built.",
        example = "false"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> autoFlushOnDelete;

    @Schema(
        title = "ID field name",
        description = "Collection field holding the document ID. Not set by default, in which case the collection schema's own field name is used.",
        example = "id"
    )
    @PluginProperty(group = "advanced")
    private Property<String> idFieldName;

    @Schema(
        title = "Text field name",
        description = "Collection field holding the original text segment. Not set by default, in which case the collection schema's own field name is used.",
        example = "text"
    )
    @PluginProperty(group = "advanced")
    private Property<String> textFieldName;

    @Schema(
        title = "Metadata field name",
        description = "Collection field holding the document metadata. Not set by default, in which case the collection schema's own field name is used.",
        example = "metadata"
    )
    @PluginProperty(group = "advanced")
    private Property<String> metadataFieldName;

    @Schema(
        title = "Vector field name",
        description = "Collection field holding the embedding vector. It must match the index definition and the embedding dimensionality. Not set by default, in which case the collection schema's own field name is used.",
        example = "vector"
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
