package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.mcp.client.DefaultMcpClient;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.service.tool.ToolExecutor;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ToolProvider;
import io.kestra.plugin.ai.tool.internal.DockerMcpTransport;
import io.kestra.plugin.scripts.runner.docker.DockerService;
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

import static io.kestra.core.utils.Rethrow.throwSupplier;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@Plugin(
    examples = {
        @Example(
            title = "Agent calling an MCP Server in a Docker container",
            full = true,
            code = {
                """
                id: docker_mcp_client
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: What is the current UTC time?

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash
                    prompt "{{ inputs.prompt }}"
                    tools:
                      - type: io.kestra.plugin.ai.tool.DockerMcpClient
                        image: mcp/time"""
            }
        ),
    }
)
@JsonDeserialize
@Schema(
    title = "Model Context Protocol (MCP) Docker client tool"
)
public class DockerMcpClient extends ToolProvider {
    @Schema(title = "The MCP client command, as a list of command parts")
    private Property<List<String>> command;

    @Schema(title = "Environment variables")
    private Property<Map<String, String>> env;

    @Schema(title = "The container image")
    @NotNull
    private Property<String> image;

    @Schema(title = "Whether to log events")
    @NotNull
    @Builder.Default
    private Property<Boolean> logEvents = Property.ofValue(false);

    @Schema(title = "The Docker host")
    private Property<String> dockerHost;

    @Schema(title = "The Docker configuration")
    private Property<String> dockerConfig;

    @Schema(title = "The Docker context")
    private Property<String> dockerContext;

    @Schema(title = "The Docker certification path")
    private Property<String> dockerCertPath;

    @Schema(title = "Whether Docker to verify TLS certificates")
    private Property<Boolean> dockerTlsVerify;

    @Schema(title = "The container registry email")
    private Property<String> registryEmail;

    @Schema(title = "The container registry password")
    private Property<String> registryPassword;

    @Schema(title = "The container registry username")
    private Property<String> registryUsername;

    @Schema(title = "The container registry URL")
    private Property<String> registryUrl;

    @Schema(title = "The API version")
    private Property<String> apiVersion;

    @JsonIgnore
    private transient McpClient mcpClient;

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext) throws IllegalVariableEvaluationException {
        String resolvedHost = runContext.render(dockerHost).as(String.class)
            .orElseGet(throwSupplier(() -> DockerService.findHost(runContext, null)));
        McpTransport transport = new DockerMcpTransport.Builder()
            .command(runContext.render(command).asList(String.class))
            .environment(runContext.render(env).asMap(String.class, String.class))
            .image(runContext.render(image).as(String.class).orElseThrow())
            .dockerHost(resolvedHost)
            .dockerConfig(runContext.render(dockerConfig).as(String.class).orElse(null))
            .dockerContext(runContext.render(dockerContext).as(String.class).orElse(null))
            .dockerCertPath(runContext.render(dockerCertPath).as(String.class).orElse(null))
            .dockerTslVerify(runContext.render(dockerTlsVerify).as(Boolean.class).orElse(null))
            .registryEmail(runContext.render(registryEmail).as(String.class).orElse(null))
            .registryPassword(runContext.render(registryUsername).as(String.class).orElse(null))
            .registryUsername(runContext.render(registryUsername).as(String.class).orElse(null))
            .registryUrl(runContext.render(registryUrl).as(String.class).orElse(null))
            .apiVersion(runContext.render(apiVersion).as(String.class).orElse(null))
            .logEvents(runContext.render(logEvents).as(Boolean.class).orElse(false))
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
