package io.kestra.plugin.ai.retriever;

import java.io.IOException;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Metric;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ContentRetrieverProvider;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;
import io.kestra.plugin.ai.domain.ModelProvider;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(
    title = "Retrieve context from an embedding store",
    description = """
        Builds a content retriever over the configured embedding store using a query embedding from `embeddingProvider`. Results are filtered by `maxResults` and `minScore` (0–1). The store is not mutated; ensure the embedding model dimension matches stored vectors."""
)
@Plugin(
    examples = {
        @Example(
            title = "Use RAG with AIAgent using an embedding store content retriever. This example ingests documents into a KV embedding store and then uses an AI agent with the EmbeddingStoreRetriever to answer questions grounded in the ingested data.",
            code = """
                id: agent_with_rag
                namespace: company.ai

                tasks:
                  - id: ingest
                    type: io.kestra.plugin.ai.rag.IngestDocument
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-001
                      googleApiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    drop: true
                    fromDocuments:
                      - content: Paris is the capital of France with a population of over 2.1 million people
                      - content: The Eiffel Tower is the most famous landmark in Paris at 330 meters tall

                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.0-flash
                      googleApiKey: "{{ secret('GEMINI_API_KEY') }}"
                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.EmbeddingStoreRetriever
                        embeddings:
                          type: io.kestra.plugin.ai.embeddings.KestraKVStore
                        embeddingProvider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-embedding-001
                          googleApiKey: "{{ secret('GEMINI_API_KEY') }}"
                        maxResults: 3
                        minScore: 0.0
                    prompt: What is the capital of France and how many people live there?
                """
        ),
        @Example(
            title = "Use multiple embedding stores simultaneously. This demonstrates the power of the content retriever approach - you can retrieve from multiple embedding stores and other sources in a single task.",
            code = """
                id: multi_store_rag
                namespace: company.ai

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.0-flash
                      googleApiKey: "{{ secret('GEMINI_API_KEY') }}"
                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.EmbeddingStoreRetriever
                        embeddings:
                          type: io.kestra.plugin.ai.embeddings.Pinecone
                          pineconeApiKey: "{{ secret('PINECONE_API_KEY') }}"
                          index: technical-docs
                        embeddingProvider:
                          type: io.kestra.plugin.ai.provider.OpenAI
                          googleApiKey: "{{ secret('OPENAI_API_KEY') }}"
                          modelName: text-embedding-3-small
                      - type: io.kestra.plugin.ai.retriever.EmbeddingStoreRetriever
                        embeddings:
                          type: io.kestra.plugin.ai.embeddings.Qdrant
                          host: localhost
                          port: 6333
                          collectionName: business-docs
                        embeddingProvider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-embedding-001
                          googleApiKey: "{{ secret('GEMINI_API_KEY') }}"
                      - type: io.kestra.plugin.ai.retriever.TavilyWebSearch
                        tavilyApiKey: "{{ secret('TAVILY_API_KEY') }}"
                    prompt: What are the latest trends in data orchestration?
                """
        )
    },
    metrics = {
        @Metric(
            name = "ai.provider.calls",
            type = Counter.TYPE,
            unit = "calls",
            description = "Number of times an embedding model is obtained from a provider, tagged by provider class name"
        ),
        @Metric(
            name = "ai.embedding.store.calls",
            type = Counter.TYPE,
            unit = "calls",
            description = "Number of times an embedding store is used, tagged by store class name"
        )
    }
)
public class EmbeddingStoreRetriever extends ContentRetrieverProvider {

    @Schema(
        title = "Embedding store",
        description = "The embedding store to retrieve relevant content from"
    )
    @NotNull
    @PluginProperty(group = "main")
    private EmbeddingStoreProvider embeddings;

    @Schema(
        title = "Embedding model provider",
        description = "Provider used to generate embeddings for the query. Must support embedding generation."
    )
    @NotNull
    @PluginProperty(group = "main")
    private ModelProvider embeddingProvider;

    @Schema(title = "Maximum number of results to return from the embedding store")
    @NotNull
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<Integer> maxResults = Property.ofValue(3);

    @Schema(
        title = "Minimum similarity score",
        description = "Only results with a similarity score ≥ minScore are returned. Range: 0.0 to 1.0 inclusive."
    )
    @NotNull
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<Double> minScore = Property.ofValue(0.0);

    @Override
    public ContentRetriever contentRetriever(RunContext runContext) throws IllegalVariableEvaluationException, IOException {
        var embeddingModel = embeddingProvider.embeddingModel(runContext);
        runContext.metric(Counter.of("ai.provider.calls", 1, "provider", embeddingProvider.getClass().getName()));
        var embeddingStore = embeddings.embeddingStore(runContext, embeddingModel.dimension(), false);
        runContext.metric(Counter.of("ai.embedding.store.calls", 1, "store", embeddings.getClass().getName()));

        return EmbeddingStoreContentRetriever.builder()
            .embeddingModel(embeddingModel)
            .embeddingStore(embeddingStore)
            .maxResults(runContext.render(this.maxResults).as(Integer.class).orElse(3))
            .minScore(runContext.render(this.minScore).as(Double.class).orElse(0.0))
            .build();
    }
}
