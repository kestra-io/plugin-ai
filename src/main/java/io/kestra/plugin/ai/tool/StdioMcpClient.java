package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.mcp.McpToolExecutor;
import dev.langchain4j.mcp.client.DefaultMcpClient;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.mcp.client.transport.stdio.StdioMcpTransport;
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
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
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
    title = "Model Context Protocol (MCP) Stdio client tool"
)
public class StdioMcpClient extends ToolProvider {
    @Schema(title = "MCP client command, as a list of command parts")
    @NotNull
    private Property<List<String>> command;

    @Schema(title = "Environment variables")
    private Property<Map<String, String>> env;

    @Schema(title = "Log events")
    @NotNull
    @Builder.Default
    private Property<Boolean> logEvents = Property.ofValue(false);

    @JsonIgnore
    private transient McpClient mcpClient;

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        McpTransport transport = new StdioMcpTransport.Builder()
            .command(runContext.render(command).asList(String.class, additionalVariables))
            .environment(runContext.render(env).asMap(String.class, String.class, additionalVariables))
            .logEvents(runContext.render(logEvents).as(Boolean.class, additionalVariables).orElse(false))
            .build();

        this.mcpClient = new DefaultMcpClient.Builder()
            .transport(transport)
            .build();

        return mcpClient.listTools().stream().collect(Collectors.toMap(
            tool -> tool,
            tool -> new McpToolExecutor(mcpClient)
        ));
    }

    @Override
    public void close(RunContext runContext) {
        if (mcpClient != null) {
            try {
                mcpClient.close();
            } catch (Exception e) {
                runContext.logger().warn("Unable to close the MCP client", e);
            }
        }
    }
}
