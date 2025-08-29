package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.annotation.JsonIgnore;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.mcp.McpToolExecutor;
import dev.langchain4j.mcp.client.DefaultMcpClient;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.service.tool.ToolExecutor;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ToolProvider;
import io.kestra.plugin.ai.tool.internal.CustomMcpLogMessageHandler;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.util.Map;
import java.util.stream.Collectors;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
abstract class AbstractMcpClient extends ToolProvider {
    @JsonIgnore
    private transient McpClient mcpClient;

    protected abstract McpTransport buildMcpTransport(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException;

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        McpTransport transport = buildMcpTransport(runContext, additionalVariables);

        this.mcpClient = new DefaultMcpClient.Builder()
            .transport(transport)
            .logHandler(new CustomMcpLogMessageHandler(runContext.logger()))
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
