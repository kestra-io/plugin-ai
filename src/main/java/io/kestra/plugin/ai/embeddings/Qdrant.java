package io.kestra.plugin.ai.embeddings;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.qdrant.QdrantEmbeddingStore;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;
import io.qdrant.client.QdrantClient;
import io.qdrant.client.QdrantGrpcClient;
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
    title = "Qdrant Embedding Store"
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Ingest documents into a Qdrant embedding store",
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
                      type: io.kestra.plugin.ai.embeddings.Qdrant
                      apiKey: "{{ kv('QDRANT_API_KEY') }}"
                      host: localhost
                      port: 6334
                      collectionName: embeddings
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        )
    },
    aliases = "io.kestra.plugin.langchain4j.embeddings.Qdrant"
)
public class Qdrant extends EmbeddingStoreProvider {

    @NotNull
    @Schema(title = "The API key")
    private Property<String> apiKey;

    @NotNull
    @Schema(title = "The database server host")
    private Property<String> host;

    @NotNull
    @Schema(title = "The database server port")
    private Property<Integer> port;

    @NotNull
    @Schema(title = "The collection name")
    private Property<String> collectionName;

    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) throws IllegalVariableEvaluationException {
        var renderedHost = runContext.render(host).as(String.class).orElseThrow();
        var renderedPort = runContext.render(port).as(Integer.class).orElseThrow();
        var renderedApiKey = runContext.render(apiKey).as(String.class).orElseThrow();

        // dimension is useless since the given embedding dimension will be used inside Qdrant

        var client = new QdrantClient(QdrantGrpcClient.newBuilder(renderedHost, renderedPort, false)
            .withApiKey(renderedApiKey)
            .build());

        var renderedCollectionName = runContext.render(collectionName).as(String.class).orElseThrow();
        var store = QdrantEmbeddingStore.builder()
            .apiKey(renderedApiKey)
            .host(renderedHost)
            .port(renderedPort)
            .collectionName(renderedCollectionName)
            .client(client)
            .build();

        if (drop) {
            store.removeAll();
        }

        return store;
    }
}
