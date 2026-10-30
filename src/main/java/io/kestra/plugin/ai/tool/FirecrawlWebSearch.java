package io.kestra.plugin.ai.tool;

import java.time.Duration;
import java.util.Map;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ToolProvider;
import io.kestra.plugin.ai.tool.internal.FirecrawlWebSearchEngines;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.WebSearchRequest;
import dev.langchain4j.web.search.WebSearchTool;
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
@Plugin(
    examples = {
        @Example(
            title = "Agent searching the web using the Firecrawl API",
            full = true,
            code = {
                """
                    id: research_agent
                    namespace: company.ai

                    inputs:
                      - id: prompt
                        type: STRING
                        defaults: What is the latest Kestra release and what new features does it include? (name 10 new features)

                    tasks:
                      - id: agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        prompt: "{{ inputs.prompt }}"
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-3.5-flash-lite
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.FirecrawlWebSearch
                            apiKey: "{{ secret('FIRECRAWL_API_KEY') }}"
                    """
            }
        ),
        @Example(
            title = "Agent reading the full content of each search result",
            full = true,
            code = {
                """
                    id: launch_post_agent
                    namespace: company.ai

                    inputs:
                      - id: prompt
                        type: STRING
                        defaults: Find the Kestra 2.0 launch blog post and summarize its main announcements in 5 bullet points.

                    tasks:
                      - id: agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        prompt: "{{ inputs.prompt }}"
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-3.5-flash-lite
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.FirecrawlWebSearch
                            apiKey: "{{ secret('FIRECRAWL_API_KEY') }}"
                            maxResults: 2
                            scrapeContent: true
                            timeBasedFilter: "qdr:m"
                    """
            }
        ),
    }
)
@JsonDeserialize
@Schema(
    title = "Search the web with Firecrawl",
    description = """
        Uses [Firecrawl Search](https://www.firecrawl.dev/search) to fetch live web results for the agent. Requires a Firecrawl API key; each search uses Firecrawl credits. Set `scrapeContent` to also return each result page as Markdown."""
)
public class FirecrawlWebSearch extends ToolProvider {
    @Schema(
        title = "Firecrawl API Key",
        description = "Firecrawl API key authorizing web-search calls, obtainable from [the Firecrawl dashboard](https://www.firecrawl.dev/app/api-keys). Store it as a Kestra secret rather than inline. No default: this property is required.",
        example = "{{ secret('FIRECRAWL_API_KEY') }}"
    )
    @NotNull
    @PluginProperty(secret = true, group = "main")
    private Property<String> apiKey;

    @Schema(
        title = "Maximum results",
        description = "Number of results returned for each search the agent runs, between 1 and 100. Defaults to `5`.",
        example = "3"
    )
    @NotNull
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<@Min(1) @Max(100) Integer> maxResults = Property.ofValue(5);

    @Schema(
        title = "Scrape result pages",
        description = "If `true`, Firecrawl also scrapes each search result and returns the page content as Markdown, so the agent reads the full page and not only a snippet. Every page goes into the model context, so keep `maxResults` low. Scraping uses additional Firecrawl credits. Defaults to `false`.",
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
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        var searchEngine = FirecrawlWebSearchEngines.buildEngine(
            runContext.render(this.apiKey).as(String.class, additionalVariables).orElse(null),
            runContext.render(this.scrapeContent).as(Boolean.class, additionalVariables).orElse(false),
            runContext.render(this.location).as(String.class, additionalVariables).orElse(null),
            runContext.render(this.timeBasedFilter).as(String.class, additionalVariables).orElse(null),
            runContext.render(this.baseUrl).as(String.class, additionalVariables).orElse(null),
            runContext.render(this.timeout).as(Duration.class, additionalVariables).orElse(null)
        );
        var rMaxResults = FirecrawlWebSearchEngines.checkMaxResults(runContext.render(this.maxResults).as(Integer.class, additionalVariables).orElse(5));

        // WebSearchTool sends no result count, so Firecrawl would return its default of 10 results per search
        WebSearchEngine cappedSearchEngine = request -> searchEngine.search(
            WebSearchRequest.builder().searchTerms(request.searchTerms()).maxResults(rMaxResults).build()
        );

        return extract(new WebSearchTool(cappedSearchEngine));
    }
}
