package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.mcp.client.transport.http.StreamableHttpMcpTransport;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.time.Duration;
import java.util.Map;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@Plugin(
    examples = {
        @Example(
            title = "Agent calling an MCP server via SSE",
            full = true,
            code = {
                """
                id: mcp_client_sse
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: Find the 2 restaurants in Lille, France with the best reviews.

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    prompt: "{{inputs.prompt}}"
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                    tools:
                      - type: io.kestra.plugin.ai.tool.StreamableHttpMcpClient
                        url: https://mcp.apify.com/?actors=compass/crawler-google-places
                        timeout: PT5M
                        headers:
                          Authorization: Bearer {{ kv('APIFY_API_TOKEN') }}"""
            }
        ),
    }
)
@JsonDeserialize
@Schema(
    title = "Call MCP server over HTTP streaming",
    description = """
        Connects to an MCP server via HTTP streaming (chunked responses) and surfaces its tools to the agent. Requires `url`; `timeout`, `headers`, `logRequests`, and `logResponses` are optional and default to provider values with logging off."""
)
public class StreamableHttpMcpClient extends AbstractMcpClient {
    @Schema(title = "URL of the MCP server")
    @NotNull
    private Property<String> url;

    @Schema(title = "Connection timeout duration")
    private Property<Duration> timeout;

    @Schema(
        title = "Custom headers",
        description = "Useful, for example, for adding authentication tokens via the `Authorization` header."
    )
    private Property<Map<String, String>> headers;

    @Schema(title = "Log requests")
    @NotNull
    @Builder.Default
    private Property<Boolean> logRequests = Property.ofValue(false);

    @Schema(title = "Log responses")
    @NotNull
    @Builder.Default
    private Property<Boolean> logResponses = Property.ofValue(false);


    @Override
    protected McpTransport buildMcpTransport(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        return new StreamableHttpMcpTransport.Builder()
            .url(runContext.render(url).as(String.class, additionalVariables).orElseThrow())
            .timeout(runContext.render(timeout).as(Duration.class, additionalVariables).orElse(null))
            .logRequests(runContext.render(logRequests).as(Boolean.class, additionalVariables).orElse(false))
            .logResponses(runContext.render(logResponses).as(Boolean.class, additionalVariables).orElse(false))
            .logger(runContext.logger())
            .customHeaders(runContext.render(headers).asMap(String.class, String.class))
            .build();
    }
}
