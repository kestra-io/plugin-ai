package io.kestra.plugin.ai.embeddings;

import com.alicloud.openservices.tablestore.SyncClient;
import com.alicloud.openservices.tablestore.model.search.FieldSchema;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.tablestore.TablestoreEmbeddingStore;
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

import java.util.List;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Store embeddings in Alibaba Tablestore",
    description = "Connects to Tablestore using access keys and writes embeddings with cosine distance. Uses the configured instance/endpoint; metadata schemas are optional. `drop=true` is not supported—manage cleanup in Tablestore."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Ingest documents into a Tablestore embedding store",
            code = """
                id: document_ingestion
                namespace: company.ai

                tasks:
                  - id: ingest
                    type: io.kestra.plugin.ai.rag.IngestDocument
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-exp-03-07
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.Tablestore
                      endpoint:  "{{ secret('TABLESTORE_ENDPOINT') }}"
                      instanceName:  "{{ secret('TABLESTORE_INSTANCE_NAME') }}"
                      accessKeyId:  "{{ secret('TABLESTORE_ACCESS_KEY_ID') }}"
                      accessKeySecret:  "{{ secret('TABLESTORE_ACCESS_KEY_SECRET') }}"
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        )
    }
)
public class Tablestore extends EmbeddingStoreProvider {

    @NotNull
    @Schema(title = "Endpoint URL", description = "The base URL for the Tablestore database endpoint.")
    private Property<String> endpoint;

    @NotNull
    @Schema(title = "Instance Name", description = "The name of the Tablestore database instance.")
    private Property<String> instanceName;

    @NotNull
    @Schema(title = "Access Key ID", description = "The access key ID used for authentication with the database.")
    private Property<String> accessKeyId;

    @NotNull
    @Schema(title = "Access Key Secret", description = "The access key secret used for authentication with the database.")
    private Property<String> accessKeySecret;

    @Schema(title = "Metadata Schema List", description = "Optional list of metadata field schemas for the collection.")
    private Property<List<FieldSchema>> metadataSchemaList;
    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) throws IllegalVariableEvaluationException {
        var rMetadataSchemaList = runContext.render(metadataSchemaList).asList(FieldSchema.class);
        return new TablestoreEmbeddingStore(
           toSyncClient(runContext), dimension, rMetadataSchemaList);
    }

    private SyncClient toSyncClient(RunContext runContext) throws IllegalVariableEvaluationException {
        return new SyncClient(
            runContext.render(endpoint).as(String.class).orElseThrow(),
            runContext.render(accessKeyId).as(String.class).orElseThrow(),
            runContext.render(accessKeySecret).as(String.class).orElseThrow(),
            runContext.render(instanceName).as(String.class).orElseThrow()
        );
    }
}
