package io.kestra.plugin.ai.mcp;

import java.time.Duration;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnore;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.Enums;
import io.kestra.plugin.ai.tool.internal.CustomMcpLogMessageHandler;

import dev.langchain4j.mcp.client.DefaultMcpClient;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.mcp.client.transport.http.HttpMcpTransport;
import dev.langchain4j.mcp.client.transport.http.StreamableHttpMcpTransport;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

/**
 * Common connection properties shared by every task that talks to an MCP server directly (as opposed to
 * {@link io.kestra.plugin.ai.domain.ToolProvider}, which exposes an MCP server's tools to an agent).
 */
@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
public abstract class AbstractMcpTask extends Task {
    @JsonIgnore
    private transient McpClient mcpClient;

    @Schema(
        title = "URL of the MCP server",
        description = "Streamable HTTP or SSE endpoint of the MCP server, matching the chosen `transport`. No default: this property is required.",
        example = "http://localhost:8080/mcp"
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<String> url;

    @Schema(
        title = "Transport",
        description = "Protocol used to reach the MCP server: `STREAMABLE_HTTP` or `SSE`. Defaults to `STREAMABLE_HTTP`.",
        example = "STREAMABLE_HTTP"
    )
    @NotNull
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<Transport> transport = Property.ofValue(Transport.STREAMABLE_HTTP);

    @Schema(
        title = "Custom headers",
        description = "Extra HTTP headers sent with every request, typically to carry an authentication token via the `Authorization` header. Not set by default.",
        example = "{Authorization: \"Bearer {{ secret('MCP_TOKEN') }}\"}"
    )
    @PluginProperty(group = "advanced")
    private Property<Map<String, String>> headers;

    @Schema(
        title = "Connection timeout duration",
        description = "Maximum time to wait for a response from the MCP server. Not set by default, in which case the underlying MCP client's own default applies and this task enforces no timeout.",
        example = "PT30S"
    )
    @PluginProperty(group = "execution")
    private Property<Duration> timeout;

    @Schema(
        title = "Log requests",
        description = "If `true`, requests sent to the MCP server are logged at INFO level. Defaults to `false`.",
        example = "true"
    )
    @NotNull
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<Boolean> logRequests = Property.ofValue(false);

    @Schema(
        title = "Log responses",
        description = "If `true`, responses received from the MCP server are logged at INFO level. Defaults to `false`.",
        example = "true"
    )
    @NotNull
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<Boolean> logResponses = Property.ofValue(false);

    @SuppressWarnings("removal") // HttpMcpTransport (legacy SSE) is deprecated for removal upstream
    protected McpClient client(RunContext runContext) throws IllegalVariableEvaluationException {
        String rUrl = runContext.render(url).as(String.class).orElseThrow();
        Duration rTimeout = runContext.render(timeout).as(Duration.class).orElse(null);
        boolean rLogRequests = runContext.render(logRequests).as(Boolean.class).orElse(false);
        boolean rLogResponses = runContext.render(logResponses).as(Boolean.class).orElse(false);
        Map<String, String> rHeaders = runContext.render(headers).asMap(String.class, String.class);

        McpTransport transport = switch (runContext.render(this.transport).as(Transport.class).orElse(Transport.STREAMABLE_HTTP)) {
            case STREAMABLE_HTTP -> new StreamableHttpMcpTransport.Builder()
                .url(rUrl)
                .timeout(rTimeout)
                .logRequests(rLogRequests)
                .logResponses(rLogResponses)
                .logger(runContext.logger())
                .customHeaders(rHeaders)
                .build();
            case SSE -> new HttpMcpTransport.Builder()
                .sseUrl(rUrl)
                .timeout(rTimeout)
                .logRequests(rLogRequests)
                .logResponses(rLogResponses)
                .logger(runContext.logger())
                .customHeaders(rHeaders)
                .build();
            case UNKNOWN -> throw new IllegalArgumentException(
                "Unsupported MCP transport. Expected one of: STREAMABLE_HTTP, SSE."
            );
        };

        this.mcpClient = new DefaultMcpClient.Builder()
            .transport(transport)
            .logHandler(new CustomMcpLogMessageHandler(runContext.logger()))
            .build();

        return this.mcpClient;
    }

    protected void killClient() {
        if (this.mcpClient != null) {
            try {
                this.mcpClient.close();
            } catch (Exception ignored) {
                // Silently ignore exceptions during kill - cleanup is best-effort
            }
        }
    }

    public enum Transport {
        STREAMABLE_HTTP,
        SSE,
        UNKNOWN;

        @JsonCreator
        public static Transport fromString(final String value) {
            return Enums.getForNameIgnoreCase(value, Transport.class, UNKNOWN);
        }
    }
}
