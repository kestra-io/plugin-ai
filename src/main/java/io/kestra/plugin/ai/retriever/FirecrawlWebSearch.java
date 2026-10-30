package io.kestra.plugin.ai.retriever;

import java.time.Duration;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ContentRetrieverProvider;
import io.kestra.plugin.ai.tool.internal.FirecrawlWebSearchEngines;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
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
    title = "Retrieve web results with Firecrawl",
    description = """
        Uses [Firecrawl Search](https://www.firecrawl.dev/search) to fetch live web context for RAG. Requires a Firecrawl API key; `maxResults` caps returned results and defaults to 3. Set `scrapeContent` to retrieve each result page as Markdown. Requests use Firecrawl credits."""
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "RAG chat with a Firecrawl web search content retriever (answers grounded in live search results)",
            code = """
                id: rag
                namespace: company.ai

                tasks:
                  - id: chat_with_rag_and_websearch_content_retriever
                    type: io.kestra.plugin.ai.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-3.5-flash-lite
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.FirecrawlWebSearch
                        apiKey: "{{ secret('FIRECRAWL_API_KEY') }}"
                    prompt: What is the latest release of Kestra?
                """
        )
    }
)
public class FirecrawlWebSearch extends ContentRetrieverProvider {
    @Schema(
        title = "API Key",
        description = "Firecrawl API key used to authenticate search requests, obtainable from [the Firecrawl dashboard](https://www.firecrawl.dev/app/api-keys). Store it as a Kestra secret rather than inline. No default: this property is required.",
        example = "{{ secret('FIRECRAWL_API_KEY') }}"
    )
    @NotNull
    @PluginProperty(secret = true, group = "main")
    private Property<String> apiKey;

    @Schema(
        title = "Maximum results",
        description = "Number of search results retrieved for each query, between 1 and 100. Defaults to `3`.",
        example = "5"
    )
    @NotNull
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<@Min(1) @Max(100) Integer> maxResults = Property.ofValue(3);

    @Schema(
        title = "Scrape result pages",
        description = "If `true`, Firecrawl also scrapes each search result and returns the page content as Markdown, so the retrieved context holds the full page and not only a snippet. Every page goes into the model context, so keep `maxResults` low. Scraping uses additional Firecrawl credits. Defaults to `false`.",
        example = "true"
    )
    @Builder.Default
    @PluginProperty(group = "processing")
    private Property<Boolean> scrapeContent = Property.ofValue(false);

    @Schema(
        title = "Location",
        description = "Location used to localize search results, for example `Germany`. No default: results are not localized.",
        example = "Germany"
    )
    @PluginProperty(group = "processing")
    private Property<String> location;

    @Schema(
        title = "Time-based filter",
        description = "Restricts results by date, sent as Firecrawl's `tbs` parameter, for example `qdr:d` (past day), `qdr:w` (past week) or `qdr:m` (past month). No default: results are not filtered by date.",
        example = "qdr:w"
    )
    @PluginProperty(group = "processing")
    private Property<String> timeBasedFilter;

    @Schema(
        title = "Base URL",
        description = "Base URL of the Firecrawl API, including the `/v2/` path. Change it only to use a [self-hosted Firecrawl](https://docs.firecrawl.dev/contributing/self-host) instance. Defaults to `https://api.firecrawl.dev/v2/`.",
        example = "http://localhost:3002/v2/"
    )
    @PluginProperty(group = "connection")
    private Property<String> baseUrl;

    @Schema(
        title = "Request timeout",
        description = "Maximum time to wait for each search request. Defaults to 10 seconds, or 90 seconds when `scrapeContent` is `true`.",
        example = "PT30S"
    )
    @PluginProperty(group = "execution")
    private Property<Duration> timeout;

    @Override
    public ContentRetriever contentRetriever(RunContext runContext) throws IllegalVariableEvaluationException {
        var searchEngine = FirecrawlWebSearchEngines.buildEngine(
            runContext.render(this.apiKey).as(String.class).orElse(null),
            runContext.render(this.scrapeContent).as(Boolean.class).orElse(false),
            runContext.render(this.location).as(String.class).orElse(null),
            runContext.render(this.timeBasedFilter).as(String.class).orElse(null),
            runContext.render(this.baseUrl).as(String.class).orElse(null),
            runContext.render(this.timeout).as(Duration.class).orElse(null)
        );
        return WebSearchContentRetriever.builder()
            .webSearchEngine(searchEngine)
            .maxResults(FirecrawlWebSearchEngines.checkMaxResults(runContext.render(this.maxResults).as(Integer.class).orElse(3)))
            .build();
    }
}
