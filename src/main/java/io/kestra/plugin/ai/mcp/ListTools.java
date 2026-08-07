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

    private static ToolDefinition toToolDefinition(ToolSpecification spec) {
        Map<String, Object> parameters = spec.parameters() == null
            ? Map.of()
            : JsonSchemaElementUtils.toMap(spec.parameters());

        return new ToolDefinition(spec.name(), spec.description(), parameters);
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "The tools exposed by the MCP server")
        private final List<ToolDefinition> tools;

        @Schema(title = "Number of tools returned")
        private final Integer count;
    }

    @Schema(title = "An MCP tool definition")
    public record ToolDefinition(
        @Schema(title = "Tool name") String name,
        @Schema(title = "Tool description") String description,
        // JSON-Schema map produced by dev.langchain4j.internal.JsonSchemaElementUtils.toMap — check that helper on any langchain4j upgrade.
        @Schema(title = "Tool argument JSON schema") Map<String, Object> parameters) {
    }
}
