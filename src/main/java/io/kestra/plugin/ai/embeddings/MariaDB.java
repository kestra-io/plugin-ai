package io.kestra.plugin.ai.embeddings;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.mariadb.DefaultMetadataStorageConfig;
import dev.langchain4j.store.embedding.mariadb.MariaDbEmbeddingStore;
import dev.langchain4j.store.embedding.mariadb.MetadataStorageConfig;
import dev.langchain4j.store.embedding.mariadb.MetadataStorageMode;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import org.apache.commons.lang3.StringUtils;

import java.io.IOException;
import java.util.List;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "MariaDB Embedding Store"
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Ingest documents into a MariaDB embedding store",
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
                      type: io.kestra.plugin.ai.embeddings.MariaDB
                      username: "{{ kv('MARIADB_USERNAME') }}"
                      password: "{{ kv('MARIADB_PASSWORD') }}"
                      databaseUrl: "{{ kv('MARIADB_DATABASE_URL') }}"
                      tableName: embeddings
                      fieldName: id
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        ),
    }
)
public class MariaDB extends EmbeddingStoreProvider {

    @NotNull
    @Schema(title = "The username")
    private Property<String> username;
    @NotNull
    @Schema(title = "The password")
    private Property<String> password;

    @NotNull
    @Schema(title = "Database URL of the MariaDB database (e.g., jdbc:mariadb://host:port/dbname)")
    private Property<String> databaseUrl;
    @NotNull
    @Schema(title = "Whether to create the table if it doesn't exist")
    private Property<Boolean> createTable;

    @NotNull
    @Schema(title = "Name of the table where embeddings will be stored")
    private Property<String> tableName;

    @NotNull
    @Schema(title = "Name of the column used as the unique ID in the database")
    private Property<String> fieldName;

    @Schema(
        title = "Metadata Column Definitions",
        description = """
              List of SQL column definitions for metadata fields (e.g., 'text TEXT', 'source TEXT').
              Required only when using COLUMN_PER_KEY storage mode.
            """
    )
    private Property<List<String>> columnDefinitions;

    @Schema(
        title = "Metadata Index Definitions",
        description = """
             List of SQL index definitions for metadata columns (e.g., 'INDEX idx_text (text)').
             Used only with COLUMN_PER_KEY storage mode.
            """
    )
    private Property<List<String>> indexes;

    @Schema(
        title = "Metadata Storage Mode",
        description = """
              Determines how metadata is stored:
                - COLUMN_PER_KEY: Use individual columns for each metadata field (requires columnDefinitions and indexes).
                - COMBINED_JSON (default): Store metadata as a JSON object in a single column.
              If columnDefinitions and indexes are provided, COLUMN_PER_KEY must be used.
            """
    )
    @Builder.Default
    private Property<String> metadataStorageMode = Property.ofValue(MetadataStorageMode.COLUMN_PER_KEY.name());

    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) throws IOException, IllegalVariableEvaluationException {
        List<String> rColumnDefinitions = runContext.render(columnDefinitions).asList(String.class);
        List<String> rIndexes = runContext.render(indexes).asList(String.class);
        String rFieldName = runContext.render(this.databaseUrl).as(String.class).orElse(StringUtils.EMPTY);
        MariaDbEmbeddingStore.Builder builder = MariaDbEmbeddingStore.builder()
            .url(runContext.render(this.databaseUrl).as(String.class).orElseThrow())
            .user(runContext.render(this.username).as(String.class).orElseThrow())
            .password(runContext.render(this.password).as(String.class).orElseThrow())
            .table(runContext.render(this.tableName).as(String.class).orElseThrow())
            .dimension(dimension)
            .createTable(runContext.render(this.createTable).as(Boolean.class).orElse(false))
            .dropTableFirst(drop);
        if (!rFieldName.isEmpty()) {
            builder.idFieldName(rFieldName);
        }
        // Add metadata config only if both columns and indexes are present
        if (!rColumnDefinitions.isEmpty() && !rIndexes.isEmpty()) {
            builder.metadataStorageConfig(toMetadataStorageConfig(runContext));
        }

        MariaDbEmbeddingStore store = builder.build();
        if (drop) {
            store.removeAll();
        }
        return store;
    }

    private MetadataStorageConfig toMetadataStorageConfig(RunContext runContext) throws IllegalVariableEvaluationException {
        List<String> rColumnDefinitions = runContext.render(columnDefinitions).asList(String.class);
        List<String> rIndexes = runContext.render(indexes).asList(String.class);
        return DefaultMetadataStorageConfig.builder()
            .storageMode(toMetadataStorageMode(runContext))
            .columnDefinitions(rColumnDefinitions)
            .indexes(rIndexes)
            .build();
    }

    private MetadataStorageMode toMetadataStorageMode(RunContext runContext) throws IllegalVariableEvaluationException {
        var rMetadataStorageMode = runContext.render(metadataStorageMode).as(String.class).orElse(MetadataStorageMode.COMBINED_JSON.name());
        return MetadataStorageMode.valueOf(rMetadataStorageMode);
    }
}
