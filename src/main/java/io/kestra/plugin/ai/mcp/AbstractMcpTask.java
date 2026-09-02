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

    @Schema(title = "URL of the MCP server", description = "The Streamable HTTP or SSE endpoint of the MCP server.")
    @NotNull
    @PluginProperty(group = "main")
    private Property<String> url;

    @Schema(title = "Transport used to connect to the MCP server")
    @NotNull
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<Transport> transport = Property.ofValue(Transport.STREAMABLE_HTTP);

    @Schema(
        title = "Custom headers",
        description = "Useful, for example, for adding authentication tokens via the `Authorization` header."
    )
    @PluginProperty(group = "advanced")
    private Property<Map<String, String>> headers;

    @Schema(title = "Connection timeout duration", description = "When not set, the underlying MCP client's default timeout applies (no timeout is enforced by this task).")
    @PluginProperty(group = "execution")
    private Property<Duration> timeout;

    @Schema(title = "Log requests")
    @NotNull
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<Boolean> logRequests = Property.ofValue(false);

    @Schema(title = "Log responses")
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
