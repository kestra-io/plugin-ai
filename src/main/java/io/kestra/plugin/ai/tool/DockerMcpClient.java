package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.mcp.client.transport.docker.DockerMcpTransport;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
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

import static io.kestra.core.utils.Rethrow.throwSupplier;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@Plugin(
    examples = {
        @Example(
            title = "Agent calling an MCP server in a Docker container",
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
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash
                    prompt: "{{ inputs.prompt }}"
                    tools:
                      - type: io.kestra.plugin.ai.tool.DockerMcpClient
                        image: mcp/time"""
            }
        ),
        @Example(
            title = "Agent calling an MCP server in a Docker container and generating output files",
            full = true,
            code = {
                """
                id: docker_mcp_client
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: Create a file 'hello.txt' with the content "Hello World" in the /tmp directory.

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash
                    prompt: "{{ inputs.prompt }}"
                    tools:
                      - type: io.kestra.plugin.ai.tool.DockerMcpClient
                        image: mcp/filesystem
                        command: ["/tmp"]
                        # Mount the container path to the task working directory to access the generated file
                        binds: ["{{workingDir}}:/tmp"]
                    outputFiles:
                      - hello.txt"""
            }
        ),
    }
)
@JsonDeserialize
@Schema(
    title = "Run MCP tools in Docker",
    description = """
        Launches an MCP server inside a Docker container and exposes its tools to the agent. Requires an `image`; optional `command`, `env`, and `binds` control the container. Docker host defaults to the detected runtime; `logEvents` defaults to false. Provide registry credentials and TLS settings when pulling from private registries."""
)
public class DockerMcpClient extends AbstractMcpClient {
    @Schema(title = "MCP client command, as a list of command parts")
    private Property<List<String>> command;

    @Schema(title = "Environment variables")
    private Property<Map<String, String>> env;

    @Schema(title = "Container image")
    @NotNull
    private Property<String> image;

    @Schema(title = "Whether to log events")
    @NotNull
    @Builder.Default
    private Property<Boolean> logEvents = Property.ofValue(false);

    @Schema(title = "Docker host")
    private Property<String> dockerHost;

    @Schema(title = "Docker configuration")
    private Property<String> dockerConfig;

    @Schema(title = "Docker context")
    private Property<String> dockerContext;

    @Schema(title = "Docker certificate path")
    private Property<String> dockerCertPath;

    @Schema(title = "Whether Docker should verify TLS certificates")
    private Property<Boolean> dockerTlsVerify;

    @Schema(title = "Container registry email")
    private Property<String> registryEmail;

    @Schema(title = "Container registry password")
    private Property<String> registryPassword;

    @Schema(title = "Container registry username")
    private Property<String> registryUsername;

    @Schema(title = "Container registry URL")
    private Property<String> registryUrl;

    @Schema(title = "API version")
    private Property<String> apiVersion;

    @Schema(title = "Volume binds")
    private Property<List<String>> binds;

    @Override
    protected McpTransport buildMcpTransport(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        String resolvedHost = runContext.render(dockerHost).as(String.class, additionalVariables)
            .orElseGet(throwSupplier(() -> DockerService.findHost(runContext, null)));
        runContext.logger().debug("Connecting to Docker host: {}", resolvedHost);

        return new DockerMcpTransport.Builder()
            .command(runContext.render(command).asList(String.class, additionalVariables))
            .environment(runContext.render(env).asMap(String.class, String.class, additionalVariables))
            .image(runContext.render(image).as(String.class, additionalVariables).orElseThrow())
            .dockerHost(resolvedHost)
            .dockerConfig(runContext.render(dockerConfig).as(String.class, additionalVariables).orElse(null))
            .dockerContext(runContext.render(dockerContext).as(String.class, additionalVariables).orElse(null))
            .dockerCertPath(runContext.render(dockerCertPath).as(String.class, additionalVariables).orElse(null))
            .dockerTslVerify(runContext.render(dockerTlsVerify).as(Boolean.class, additionalVariables).orElse(null))
            .registryEmail(runContext.render(registryEmail).as(String.class, additionalVariables).orElse(null))
            .registryPassword(runContext.render(registryPassword).as(String.class, additionalVariables).orElse(null))
            .registryUsername(runContext.render(registryUsername).as(String.class, additionalVariables).orElse(null))
            .registryUrl(runContext.render(registryUrl).as(String.class, additionalVariables).orElse(null))
            .apiVersion(runContext.render(apiVersion).as(String.class, additionalVariables).orElse(null))
            .logEvents(runContext.render(logEvents).as(Boolean.class, additionalVariables).orElse(false))
            .logger(runContext.logger())
            .binds(runContext.render(binds).asList(String.class, additionalVariables))
            .build();
    }
}
