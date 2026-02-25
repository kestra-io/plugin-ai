package io.kestra.plugin.ai.embeddings;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
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

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Store embeddings in Chroma",
    description = "Connects to a Chroma HTTP instance using cosine distance; targets the given collection and drops it when `drop=true`."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Ingest documents into a Chroma embedding store",
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
                      type: io.kestra.plugin.ai.embeddings.Chroma
                      baseUrl: http://localhost:8000
                      collectionName: embeddings
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        )
    },
    aliases = "io.kestra.plugin.langchain4j.embeddings.Chroma"
)
public class Chroma extends EmbeddingStoreProvider {

    @NotNull
    @Schema(title = "The database base URL")
    private Property<String> baseUrl;

    @NotNull
    @Schema(title = "The collection name")
    private Property<String> collectionName;

    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) throws IllegalVariableEvaluationException {

        // dimension is useless since the given embedding dimension will be used inside Chroma

        var renderedCollectionName = runContext.render(collectionName).as(String.class).orElseThrow();
        var store = ChromaEmbeddingStore.builder()
            .baseUrl(runContext.render(baseUrl).as(String.class).orElseThrow())
            .collectionName(renderedCollectionName)
            .build();

        if (drop) {
            store.removeAll();
        }

        return store;
    }
}
