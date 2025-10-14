package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agentic.AgenticServices;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.service.tool.ToolExecutor;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.ai.domain.ToolProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.util.Map;

@Getter
@SuperBuilder
@NoArgsConstructor
@Plugin(
    examples =  {
        @Example(
            title = "Call a remote AI agent via the A2A protocol.",
            full = true,
            code = {
                """
                    id: ai-agent-with-agent-tools
                    namespace: company.ai

                    inputs:
                      - id: prompt
                        type: STRING
                        defaults: |
                          Each flow can produce outputs that can be consumed by other flows. This is a list property, so that your flow can produce as many outputs as you need.
                          Each output needs to have an ID (the name of the output), a type (the same types you know from inputs, e.g., STRING, URI, or JSON), and a value, which is the actual output value that will be stored in internal storage and passed to other flows when needed.
                    tasks:
                      - id: ai-agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-2.5-flash
                          apiKey: "{{ kv('GEMINI_API_KEY') }}"
                        systemMessage: Summarize the user message, then translate it into French using the provided tool.
                        prompt: "{{inputs.prompt}}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.A2AAgent
                            description: Translation expert
                            serverUrl: "http://localhost:10000\""""
            }
        ),
    }
)
@JsonDeserialize
@Schema(
    title = "Call a remote AI agent via the A2A protocol.",
    description = """
        This tool allows an LLM to call a remote AI Agent via the A2A protocol.
        Make sure to specify a name and a description so the LLM can understand what it does to decide if it needs to call it."""
)
public class A2AAgent extends ToolProvider {
    private static final String TOOL_DESCRIPTION = "This tool allows to call a remote A2A agent named '%s'.";

    @Schema(
        title = "Agent name",
        description = "It must be set to a different value than the default in case you want to have multiple agents used as tools in the same task."
    )
    @NotNull
    @Builder.Default
    protected Property<String> name = Property.ofValue("tool");

    @Schema(
        title = "Agent description",
        description = "The description will be used to instruct the LLM what the tool is doing."
    )
    @NotNull
    protected Property<String> description;

    @Schema(title = "Server URL", description = "The URL of the remote agent A2A server")
    @NotNull
    protected Property<String> serverUrl;

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        String rServerUrl = runContext.render(serverUrl).as(String.class).orElseThrow();
        AgentTool agent = AgenticServices.a2aBuilder(rServerUrl, AgentTool.class)
            .build();

        var jsonSchema = JsonObjectSchema.builder()
            .description(runContext.render(this.description).as(String.class).orElseThrow())
            .addStringProperty("prompt", "The A2A agent prompt also called a user message")
            .build();

        var rName = runContext.render(this.name).as(String.class).orElseThrow();
        return Map.of(
            ToolSpecification.builder()
                .name("kestra_a2a_agent_" + rName)
                .description(TOOL_DESCRIPTION.formatted(rName))
                .parameters(jsonSchema)
                .build(),
            new AgentToolExecutor(runContext, agent)
        );
    }

    interface AgentTool {
        String invoke(String userMessage);
    }

    private static class AgentToolExecutor implements ToolExecutor {
        private final RunContext runContext;
        private final AgentTool agent;

        AgentToolExecutor(RunContext runContext, AgentTool agent) {
            this.runContext = runContext;
            this.agent = agent;
        }

        @Override
        public String execute(ToolExecutionRequest toolExecutionRequest, Object memoryId) {
            runContext.logger().debug("Tool execution request: {}", toolExecutionRequest);
            try {
                var parameters = JacksonMapper.toMap(toolExecutionRequest.arguments());
                var prompt = (String) parameters.get("prompt");

                String completion = agent.invoke(prompt);
                runContext.logger().debug("Generated completion: {}", completion);

                return completion;
            } catch (Exception e) {
                throw new ToolExecutionException(e);
            }
        }
    }
}
