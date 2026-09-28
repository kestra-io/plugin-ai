package io.kestra.plugin.ai;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

import org.junit.jupiter.api.extension.AfterAllCallback;
import org.junit.jupiter.api.extension.BeforeAllCallback;
import org.junit.jupiter.api.extension.BeforeEachCallback;
import org.junit.jupiter.api.extension.ExtensionContext;

import com.fasterxml.jackson.databind.JsonNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import io.kestra.core.serializers.JacksonMapper;

/**
 * A scripted OpenAI-compatible chat completion endpoint, so tests exercising the plugin's LLM plumbing
 * (tool calling, memory, guardrails, observability) don't depend on a public, rate-limited model.
 * <p>
 * Each request is answered by the first matching rule:
 * <ol>
 * <li>the last message is a tool result: answer with {@link #finalAnswer(Function)} applied to it (echoes it by default);</li>
 * <li>a tool call is scripted and the request offers tools: call it;</li>
 * <li>otherwise answer with {@link #reply(Function)} applied to the request messages.</li>
 * </ol>
 * The script is reset before each test.
 */
public class MockOpenAI implements BeforeAllCallback, AfterAllCallback, BeforeEachCallback {
    private static final String MODEL_NAME = "gpt-4o-mini";
    private static final String DEFAULT_REPLY = "Kestra is an open-source orchestration platform.";

    private HttpServer server;
    private final AtomicInteger requestCount = new AtomicInteger();
    private volatile Function<JsonNode, String> reply;
    private volatile Function<String, String> finalAnswer;
    private volatile String toolName;
    private volatile String toolArguments;

    @Override
    public void beforeAll(ExtensionContext context) throws IOException {
        server = HttpServer.create(new InetSocketAddress("localhost", 0), 0);
        server.createContext("/v1/chat/completions", this::handle);
        server.start();
    }

    @Override
    public void afterAll(ExtensionContext context) {
        server.stop(0);
    }

    @Override
    public void beforeEach(ExtensionContext context) {
        requestCount.set(0);
        reply = messages -> DEFAULT_REPLY;
        finalAnswer = toolResult -> toolResult;
        toolName = null;
        toolArguments = null;
    }

    public String baseUrl() {
        return "http://localhost:" + server.getAddress().getPort() + "/v1";
    }

    /** Answers requests that neither return a tool result nor trigger the scripted tool call; receives the request messages. */
    public MockOpenAI reply(Function<JsonNode, String> reply) {
        this.reply = reply;
        return this;
    }

    /** Calls the given tool with the given JSON arguments when the request offers tools. */
    public MockOpenAI callTool(String name, String arguments) {
        this.toolName = name;
        this.toolArguments = arguments;
        return this;
    }

    /** Answers a request whose last message is a tool result; receives that result. */
    public MockOpenAI finalAnswer(Function<String, String> finalAnswer) {
        this.finalAnswer = finalAnswer;
        return this;
    }

    private void handle(HttpExchange exchange) throws IOException {
        var request = JacksonMapper.ofJson().readTree(exchange.getRequestBody());
        var requestId = requestCount.incrementAndGet();

        var messages = request.path("messages");
        var lastMessage = messages.get(messages.size() - 1);
        Map<String, Object> message;
        String finishReason;
        if ("tool".equals(lastMessage.path("role").asText())) {
            message = Map.of("role", "assistant", "content", finalAnswer.apply(lastMessage.path("content").asText()));
            finishReason = "stop";
        } else if (toolName != null && !request.path("tools").isEmpty()) {
            message = Map.of(
                "role", "assistant",
                "tool_calls", List.of(
                    Map.of(
                        "id", "call_" + requestId,
                        "type", "function",
                        "function", Map.of("name", toolName, "arguments", toolArguments)
                    )
                )
            );
            finishReason = "tool_calls";
        } else {
            message = Map.of("role", "assistant", "content", reply.apply(messages));
            finishReason = "stop";
        }

        var response = Map.of(
            "id", "chatcmpl-mock-" + requestId,
            "object", "chat.completion",
            "created", 0,
            "model", MODEL_NAME,
            "choices", List.of(Map.of("index", 0, "message", message, "finish_reason", finishReason)),
            "usage", Map.of("prompt_tokens", 10, "completion_tokens", 10, "total_tokens", 20)
        );
        var body = JacksonMapper.ofJson().writeValueAsString(response).getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, body.length);
        try (var os = exchange.getResponseBody()) {
            os.write(body);
        }
    }
}
