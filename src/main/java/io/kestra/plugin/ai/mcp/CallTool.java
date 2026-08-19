package io.kestra.plugin.ai.mcp;

import java.util.Map;

import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Metric;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.service.tool.ToolExecutionResult;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Call a tool on an MCP server and return its result",
    description = """
        Connects to a Model Context Protocol (MCP) server, invokes one of its tools with the given arguments, and returns the result — no LLM involved. Use `io.kestra.plugin.ai.mcp.ListTools` to discover the available tools and their argument schemas first."""
)
@Plugin(
    examples = {
        @Example(
            title = "Call the `add` tool on an MCP server over Streamable HTTP.",
            full = true,
            code = """
                id: mcp_call_tool
                namespace: company.ai

                tasks:
                  - id: call
                    type: io.kestra.plugin.ai.mcp.CallTool
                    url: https://mcp.example.com/mcp
                    headers:
                      Authorization: "Bearer {{ secret('MCP_API_TOKEN') }}"
                    tool: add
                    arguments:
                      a: 5
                      b: 12
                """
        )
    },
    metrics = {
        @Metric(
            name = "mcp.tool.calls",
            type = Counter.TYPE,
            unit = "calls",
            description = "Number of MCP tool calls, tagged by tool name"
        )
    }
)
public class CallTool extends AbstractMcpTask implements RunnableTask<CallTool.Output> {
    @Schema(title = "Name of the tool to call")
    @NotNull
    @PluginProperty(group = "main")
    private Property<String> tool;

    @Schema(title = "Arguments passed to the tool")
    @PluginProperty(group = "main")
    private Property<Map<String, Object>> arguments;

    @Schema(
        title = "Fail on tool error",
        description = "Whether the task should fail when the MCP server reports the tool call as an error. When `false`, the error is instead reported in `errorMessage` and `isError`."
    )
    @NotNull
    @Builder.Default
    @PluginProperty(group = "advanced")
    private Property<Boolean> failOnToolError = Property.ofValue(true);

    @Override
    public Output run(RunContext runContext) throws Exception {
        String rTool = runContext.render(this.tool).as(String.class).orElseThrow();
        Map<String, Object> rArguments = runContext.render(this.arguments).asMap(String.class, Object.class);
        boolean rFailOnToolError = runContext.render(this.failOnToolError).as(Boolean.class).orElse(true);

        ToolExecutionRequest request = ToolExecutionRequest.builder()
            .name(rTool)
            .arguments(JacksonMapper.ofJson().writeValueAsString(rArguments))
            .build();

        runContext.metric(Counter.of("mcp.tool.calls", 1, "tool", rTool));

        try (McpClient client = client(runContext)) {
            ToolExecutionResult result = client.executeTool(request);

            if (result.isError() && rFailOnToolError) {
                throw new IllegalStateException("MCP tool '" + rTool + "' returned an error: " + result.resultText());
            }

            if (result.isError()) {
                runContext.logger().warn("MCP tool '{}' returned an error: {}", rTool, result.resultText());
            }

            return Output.builder()
                .result(result.resultText())
                .structuredContent(result.result())
                .isError(result.isError())
                .errorMessage(result.isError() ? result.resultText() : null)
                .build();
        } catch (ToolExecutionException | ToolArgumentsException e) {
            if (rFailOnToolError) {
                throw e;
            }

            runContext.logger().warn("MCP tool '{}' failed: {}", rTool, e.getMessage());
            return Output.builder()
                .isError(true)
                .errorMessage(e.getMessage())
                .build();
        }
    }

    @Override
    public void kill() {
        killClient();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Tool result", description = "The tool's text result.")
        private final String result;

        @Schema(title = "Structured content", description = "The tool's structured content, when the server returned one.")
        private final Object structuredContent;

        @Schema(title = "Whether the tool call returned an error")
        private final Boolean isError;

        @Schema(title = "Error message", description = "Populated only when `failOnToolError` is `false` and the call failed.")
        private final String errorMessage;
    }
}
