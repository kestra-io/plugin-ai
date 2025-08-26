package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.WebSearchTool;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ToolProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.util.Map;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@Plugin(
    examples = {
        @Example(
            title = "Agent searching the web using the Tavily API",
            full = true,
            code = {
                """
                id: research_agent
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: What is the latest Kestra release and what new features does it include?

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    prompt: "{{ inputs.prompt }}"
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    tools:
                      - type: io.kestra.plugin.ai.tool.TavilyWebSearchTool
                        apiKey: "{{ secret('TAVILY_API_KEY') }}"
                """
            }
        ),
    },
    aliases = "io.kestra.plugin.langchain4j.tool.TavilyWebSearch"
)
@JsonDeserialize
@Schema(
    title = "WebSearch tool for Tavily Search"
)
public class TavilyWebSearch extends ToolProvider {
    @Schema(title = "Tavily API Key - you can obtain one from [the Tavily website](https://www.tavily.com/#pricing)")
    @NotNull
    private Property<String> apiKey;

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        final WebSearchEngine searchEngine = TavilyWebSearchEngine.builder()
            .apiKey(runContext.render(this.apiKey).as(String.class, additionalVariables).orElseThrow())
            .build();

        return extract(new WebSearchTool(searchEngine));
    }
}
