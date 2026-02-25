package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.code.judge0.Judge0JavaScriptExecutionTool;
import dev.langchain4j.service.tool.ToolExecutor;
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
            title = "Agent performing mathematical calculations using the Judge0 Code Execution API",
            full = true,
            code = """
                id: calculator_agent
                namespace: company.ai

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash
                    prompt: What is the square root of 49506838032859?
                    tools:
                      - type: io.kestra.plugin.ai.tool.CodeExecution
                        apiKey: "{{ secret('RAPID_API_KEY') }}"
                """
        ),
    }
)
@JsonDeserialize
@Schema(
    title = "Execute JavaScript via Judge0",
    description = """
        Sends JavaScript snippets to the Judge0 sandbox (RapidAPI) and returns the program output. Requires a RapidAPI key; execution limits and timeouts follow the Judge0 plan. Avoid sending untrusted secrets in code."""
)
public class CodeExecution extends ToolProvider {

    @Schema(
        title = "RapidAPI key for Judge0",
        description = "You can obtain it from the [RapidAPI website](https://rapidapi.com/judge0-official/api/judge0-ce/pricing)."
    )
    @NotNull
    private Property<String> apiKey;

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        return extract(new Judge0JavaScriptExecutionTool(runContext.render(this.apiKey).as(String.class, additionalVariables).orElseThrow()));
    }
}
