package io.kestra.plugin.ai.tool;

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.scripts.runner.docker.DockerService;

import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.mcp.client.transport.docker.DockerMcpTransport;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import static io.kestra.core.utils.Rethrow.throwSupplier;
import io.kestra.core.models.annotations.PluginProperty;

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
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                          modelName: gemini-3.5-flash-lite
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
                        defaults: Create the file '/tmp/hello.txt' with the content "Hello World".

                    tasks:
                      - id: agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                          modelName: gemini-3.5-flash-lite
                        prompt: "{{ inputs.prompt }}"
                        systemMessage: |
                          You are a filesystem assistant. Always use the write_file tool with the exact absolute path provided in the user's request.
                        tools:
                          - type: io.kestra.plugin.ai.tool.DockerMcpClient
                            image: mcp/filesystem
                            command: ["/tmp"]
                            # Mount the container path to the task working directory to access the generated file
                            binds: ["{{ workingDir }}:/tmp"]
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
    @Schema(
        title = "MCP server arguments",
        description = "Arguments passed to the container entrypoint, each element a separate command part. Not set by default, in which case the image's own entrypoint arguments are used.",
        example = "[\"/tmp\"]"
    )
    @PluginProperty(group = "advanced")
    private Property<List<String>> command;

    @Schema(
        title = "Environment variables",
        description = "Environment variables set inside the container, typically to supply credentials to the MCP server. Not set by default.",
        example = "{GITHUB_PERSONAL_ACCESS_TOKEN: \"{{ secret('GITHUB_TOKEN') }}\"}"
    )
    @PluginProperty(group = "execution")
    private Property<Map<String, String>> env;

    @Schema(
        title = "Container image",
        description = "Docker image running the MCP server. No default: this property is required.",
        example = "mcp/filesystem"
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<String> image;

    @Schema(
        title = "Log events",
        description = "If `true`, MCP protocol events exchanged with the container are logged. Defaults to `false`.",
        example = "true"
    )
    @NotNull
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<Boolean> logEvents = Property.ofValue(false);

    @Schema(
        title = "Docker host",
        description = "URI of the Docker daemon that runs the container. Not set by default, in which case the host is auto-detected from the worker environment.",
        example = "unix:///var/run/docker.sock"
    )
    @PluginProperty(group = "connection")
    private Property<String> dockerHost;

    @Schema(
        title = "Docker configuration",
        description = "Docker client configuration as JSON, typically holding registry credentials. Not set by default, in which case the worker's Docker config is used.",
        example = "{{ secret('DOCKER_CONFIG') }}"
    )
    @PluginProperty(group = "advanced")
    private Property<String> dockerConfig;

    @Schema(
        title = "Docker context",
        description = "Name of the Docker CLI context selecting which daemon to talk to. Not set by default (the current context is used).",
        example = "default"
    )
    @PluginProperty(group = "advanced")
    private Property<String> dockerContext;

    @Schema(
        title = "Docker certificate path",
        description = "Directory holding the TLS client certificates used to reach the Docker daemon. Not set by default.",
        example = "/home/kestra/.docker/certs"
    )
    @PluginProperty(group = "advanced")
    private Property<String> dockerCertPath;

    @Schema(
        title = "Verify Docker TLS certificates",
        description = "If `true`, verify the Docker daemon's TLS certificate when connecting over TLS. Not set by default, in which case the Docker client default applies.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> dockerTlsVerify;

    @Schema(
        title = "Container registry email",
        description = "Email associated with the container registry account, required by some private registries. Not set by default.",
        example = "user@example.com"
    )
    @PluginProperty(group = "advanced")
    private Property<String> registryEmail;

    @Schema(
        title = "Container registry password",
        description = "Password or token used to pull the image from a private registry. Store it as a Kestra secret rather than inline. Not set by default (anonymous pull).",
        example = "{{ secret('REGISTRY_PASSWORD') }}"
    )
    @PluginProperty(secret = true, group = "connection")
    private Property<String> registryPassword;

    @Schema(
        title = "Container registry username",
        description = "User authenticating against a private container registry. Not set by default (anonymous pull).",
        example = "registry_user"
    )
    @PluginProperty(group = "connection")
    private Property<String> registryUsername;

    @Schema(
        title = "Container registry URL",
        description = "Registry the image is pulled from. Not set by default, in which case Docker Hub is used.",
        example = "https://index.docker.io/v1/"
    )
    @PluginProperty(group = "connection")
    private Property<String> registryUrl;

    @Schema(
        title = "Docker API version",
        description = "Docker Engine API version used by the client. Not set by default, in which case the version is negotiated with the daemon.",
        example = "1.44"
    )
    @PluginProperty(group = "advanced")
    private Property<String> apiVersion;

    @Schema(
        title = "Volume binds",
        description = "Host-to-container volume mounts in `host_path:container_path` form, used for example to share the task working directory with the MCP server. Not set by default (no mount).",
        example = "[\"{{ workingDir }}:/tmp\"]"
    )
    @PluginProperty(group = "advanced")
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
