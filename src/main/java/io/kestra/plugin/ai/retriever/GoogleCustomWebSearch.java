package io.kestra.plugin.ai.retriever;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.google.customsearch.GoogleCustomWebSearchEngine;
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
    title = "Web search content retriever for Google Custom Search"
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "RAG chat with a web search content retriever (answers grounded in search results)",
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
                      - type: io.kestra.plugin.ai.retriever.GoogleCustomWebSearch
                        apiKey: "{{ kv('GOOGLE_SEARCH_API_KEY') }}"
                        csi: "{{ kv('GOOGLE_SEARCH_CSI') }}"
                    prompt: What is the latest release of Kestra?
                """
        )
    },
    aliases = "io.kestra.plugin.langchain4j.retriever.GoogleCustomWebSearch"
)
public class GoogleCustomWebSearch extends ContentRetrieverProvider {
    @Schema(title = "Custom search engine ID (cx)")
    @NotNull
    private Property<String> csi;

    @Schema(title = "API key")
    @NotNull
    private Property<String> apiKey;

    @Schema(title = "Maximum number of results")
    @NotNull
    @Builder.Default
    private Property<Integer> maxResults = Property.ofValue(3);

    @Override
    public ContentRetriever contentRetriever(RunContext runContext) throws IllegalVariableEvaluationException {
        final WebSearchEngine searchEngine = GoogleCustomWebSearchEngine.builder()
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .csi((runContext.render(this.csi).as(String.class).orElseThrow()))
            .build();
        return WebSearchContentRetriever.builder()
            .webSearchEngine(searchEngine)
            .maxResults(runContext.render(this.maxResults).as(Integer.class).orElse(3))
            .build();
    }
}
