package io.kestra.plugin.ai.mcp;

import java.util.List;
import java.util.Map;

import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.internal.JsonSchemaElementUtils;
import dev.langchain4j.mcp.client.McpClient;
import io.swagger.v3.oas.annotations.media.Schema;
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
    title = "List the tools exposed by an MCP server",
    description = """
        Connects to a Model Context Protocol (MCP) server and returns its tool catalogue — name, description and argument JSON schema for each tool. Use `io.kestra.plugin.ai.mcp.CallTool` to invoke one of them."""
)
@Plugin(
    examples = {
        @Example(
            title = "Discover the tools available on an MCP server over Streamable HTTP.",
            full = true,
            code = """
                id: mcp_list_tools
                namespace: company.ai

                tasks:
                  - id: list
                    type: io.kestra.plugin.ai.mcp.ListTools
                    url: https://mcp.example.com/mcp
                """
        )
    }
)
public class ListTools extends AbstractMcpTask implements RunnableTask<ListTools.Output> {
    @Override
    public Output run(RunContext runContext) throws Exception {
        try (McpClient client = client(runContext)) {
            List<ToolDefinition> rTools = client.listTools().stream()
                .map(ListTools::toToolDefinition)
                .toList();

            return Output.builder()
                .tools(rTools)
                .count(rTools.size())
                .build();
        }
    }

    @Override
    public void kill() {
        killClient();
    }

    private static ToolDefinition toToolDefinition(ToolSpecification spec) {
        Map<String, Object> parameters = spec.parameters() == null
            ? Map.of()
            : JsonSchemaElementUtils.toMap(spec.parameters());

        return new ToolDefinition(spec.name(), spec.description(), parameters);
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(
            title = "Tools",
            description = "Tools the MCP server exposes, each with its name, description and input schema.",
            example = "[{name: \"get_current_time\", description: \"Returns the current time in a timezone\"}]"
        )
        private final List<ToolDefinition> tools;

        @Schema(
            title = "Tool count",
            description = "Number of tools returned by the server.",
            example = "3"
        )
        private final Integer count;
    }

    @Schema(title = "An MCP tool definition")
    public record ToolDefinition(
        @Schema(
            title = "Tool name",
            description = "Name the tool is invoked by, to pass as `tool` on an `io.kestra.plugin.ai.mcp.CallTool` task.",
            example = "get_current_time"
        ) String name,
        @Schema(
            title = "Tool description",
            description = "Natural-language summary the server publishes for this tool, which an LLM uses to decide when to call it.",
            example = "Returns the current time in a given timezone."
        ) String description,
        // JSON-Schema map produced by dev.langchain4j.internal.JsonSchemaElementUtils.toMap — check that helper on any langchain4j upgrade.
        @Schema(
            title = "Tool argument JSON schema",
            description = "JSON Schema describing the arguments the tool accepts, which `CallTool` arguments must conform to.",
            example = "{\"type\": \"object\", \"properties\": {\"timezone\": {\"type\": \"string\"}}}"
        ) Map<String, Object> parameters) {
    }
}
