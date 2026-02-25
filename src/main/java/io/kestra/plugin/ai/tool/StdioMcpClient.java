package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.mcp.client.transport.stdio.StdioMcpTransport;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.util.List;
import java.util.Map;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@Plugin(
    examples = {
        @Example(
            title = "Agent calling an MCP server via Stdio",
            full = true,
            code = {
                """
                id: mcp_client_stdio
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: What is the current time in New York?

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    prompt: "{{ inputs.prompt }}"
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash
                    tools:
                      - type: io.kestra.plugin.ai.tool.StdioMcpClient
                        command: ["docker", "run", "--rm", "-i", "mcp/time"]
                """
            }
        ),
    },
    aliases = "io.kestra.plugin.langchain4j.tool.StdioMcpClient"
)
@JsonDeserialize
@Schema(
    title = "Run MCP tools over stdio",
    description = """
        Starts an MCP server via a local command and exposes its advertised tools to the agent over stdio. `command` is required; `logEvents` defaults to false. Use `env` to pass credentials or config needed by the server process."""
)
public class StdioMcpClient extends AbstractMcpClient {
    @Schema(title = "MCP client command, as a list of command parts")
    @NotNull
    private Property<List<String>> command;

    @Schema(title = "Environment variables")
    private Property<Map<String, String>> env;

    @Schema(title = "Log events")
    @NotNull
    @Builder.Default
    private Property<Boolean> logEvents = Property.ofValue(false);


    @Override
    protected McpTransport buildMcpTransport(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        return new StdioMcpTransport.Builder()
            .command(runContext.render(command).asList(String.class, additionalVariables))
            .environment(runContext.render(env).asMap(String.class, String.class, additionalVariables))
            .logEvents(runContext.render(logEvents).as(Boolean.class, additionalVariables).orElse(false))
            .build();
    }

}
