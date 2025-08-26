package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.WebSearchTool;
import dev.langchain4j.web.search.google.customsearch.GoogleCustomWebSearchEngine;
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
    examples =  {
        @Example(
            title = "Agent searching the web with Google Custom Search",
            full = true,
            code = {
                """
                id: agent_searching_web
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: What is the latest Kestra release and what new features does it include?

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash
                    prompt: "{{ inputs.prompt }}"
                    tools:
                      - type: io.kestra.plugin.ai.tool.GoogleCustomWebSearch
                        apiKey: "{{ secret('GOOGLE_SEARCH_API_KEY') }}"
                        csi: "{{ secret('GOOGLE_SEARCH_CSI') }}"
                """
            }
        ),
    },
    aliases = "io.kestra.plugin.langchain4j.tool.GoogleCustomWebSearch"
)
@JsonDeserialize
@Schema(
    title = "WebSearch tool for Google Custom Search"
)
public class GoogleCustomWebSearch extends ToolProvider {
    @Schema(title = "API Key")
    @NotNull
    private Property<String> csi;

    @Schema(title = "API Key")
    @NotNull
    private Property<String> apiKey;

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        final WebSearchEngine searchEngine = GoogleCustomWebSearchEngine.builder()
            .apiKey(runContext.render(this.apiKey).as(String.class, additionalVariables).orElseThrow())
            .csi((runContext.render(this.csi).as(String.class, additionalVariables).orElseThrow()))
            .build();

        return extract(new WebSearchTool(searchEngine));
    }
}
