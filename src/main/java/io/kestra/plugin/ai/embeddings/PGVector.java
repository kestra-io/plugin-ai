package io.kestra.plugin.ai.embeddings;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import io.kestra.core.models.annotations.PluginProperty;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Store embeddings with pgvector",
    description = "Uses the PostgreSQL pgvector extension to persist embeddings in the given table. `drop=true` recreates the table; optional IVF index (`useIndex`) defaults to false. Ensure pgvector extension is installed and the user can create tables/indexes."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Ingest documents into a PGVector embedding store",
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
                      type: io.kestra.plugin.ai.embeddings.PGVector
                      host: localhost
                      port: 5432
                      user: "{{ secret('POSTGRES_USER') }}"
                      password: "{{ secret('POSTGRES_PASSWORD') }}"
                      database: postgres
                      table: embeddings
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        )
    },
    aliases = "io.kestra.plugin.langchain4j.embeddings.PGVector"
)
public class PGVector extends EmbeddingStoreProvider {
    @NotNull
    @Schema(
        title = "Database server host",
        description = "Hostname or IP address of the PostgreSQL server running the pgvector extension. No default: this property is required.",
        example = "localhost"
    )
    @PluginProperty(group = "main")
    private Property<String> host;

    @NotNull
    @Schema(
        title = "Database server port",
        description = "TCP port the PostgreSQL server listens on. No default: this property is required.",
        example = "5432"
    )
    @PluginProperty(group = "main")
    private Property<Integer> port;

    @NotNull
    @Schema(
        title = "Database user",
        description = "User connecting to the PostgreSQL database. No default: this property is required.",
        example = "postgres"
    )
    @PluginProperty(group = "main")
    private Property<String> user;

    @NotNull
    @Schema(
        title = "Database password",
        description = "Password of the database user. Store it as a Kestra secret rather than inline. No default: this property is required.",
        example = "{{ secret('PGVECTOR_PASSWORD') }}"
    )
    @PluginProperty(secret = true, group = "main")
    private Property<String> password;

    @NotNull
    @Schema(
        title = "Database name",
        description = "Name of the PostgreSQL database holding the embeddings table. No default: this property is required.",
        example = "vectordb"
    )
    @PluginProperty(group = "main")
    private Property<String> database;

    @NotNull
    @Schema(
        title = "Table name",
        description = "Table that stores the embeddings. No default: this property is required. When `drop` is requested by the ingestion task, the table is recreated.",
        example = "embeddings"
    )
    @PluginProperty(group = "main")
    private Property<String> table;

    @Schema(
        title = "Use an IVFFlat index",
        description = "If `true`, build an IVFFlat index on the embedding column. IVFFlat divides vectors into lists and searches only the lists closest to the query vector: it builds faster and uses less memory than HNSW, at the cost of a worse speed-recall tradeoff. Defaults to `false`.",
        example = "true"
    )
    @Builder.Default
    @PluginProperty(group = "advanced")
    private Property<Boolean> useIndex = Property.ofValue(false);

    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) throws IllegalVariableEvaluationException {
        return PgVectorEmbeddingStore.builder()
            .host(runContext.render(host).as(String.class).orElseThrow())
            .port(runContext.render(port).as(Integer.class).orElseThrow())
            .database(runContext.render(database).as(String.class).orElseThrow())
            .user(runContext.render(user).as(String.class).orElseThrow())
            .password(runContext.render(password).as(String.class).orElseThrow())
            .table(runContext.render(table).as(String.class).orElseThrow())
            .dropTableFirst(drop)
            .dimension(dimension)
            .useIndex(runContext.render(useIndex).as(Boolean.class).orElseThrow())
            .build();
    }
}
