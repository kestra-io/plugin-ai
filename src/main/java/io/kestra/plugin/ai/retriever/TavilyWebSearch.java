package io.kestra.plugin.ai.retriever;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ContentRetrieverProvider;
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
    title = "WebSearch content retriever for Tavily Search"
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Chat with your data using Retrieval Augmented Generation (RAG) and a WebSearch content retriever. The Chat with RAG retrieves contents from a WebSearch client and provides a response grounded in data rather than hallucinating.",
            code = """
                id: rag
                namespace: company.ai

                tasks:
                  - id: chat_with_rag_and_websearch_content_retriever
                    type: io.kestra.plugin.ai.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.TavilyWebSearch
                        apiKey: "{{ kv('TAVILY_API_KEY') }}"
                    prompt: What is the latest release of Kestra?
                """
        )
    },
    aliases = "io.kestra.plugin.langchain4j.retriever.TavilyWebSearch"
)
public class TavilyWebSearch extends ContentRetrieverProvider {
    @Schema(title = "API Key")
    @NotNull
    private Property<String> apiKey;

    @Schema(title = "Maximum number of results to return")
    @NotNull
    @Builder.Default
    private Property<Integer> maxResults = Property.ofValue(3);

    @Override
    public ContentRetriever contentRetriever(RunContext runContext) throws IllegalVariableEvaluationException {
        final WebSearchEngine searchEngine = TavilyWebSearchEngine.builder()
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .build();
        return WebSearchContentRetriever.builder()
            .webSearchEngine(searchEngine)
            .maxResults(runContext.render(this.maxResults).as(Integer.class).orElse(3))
            .build();
    }
}
