package io.kestra.plugin.ai.provider;

import java.util.List;
import java.util.Map;
import java.util.Objects;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.parallel.ResourceLock;

import com.github.tomakehurst.wiremock.junit5.WireMockExtension;
import com.github.tomakehurst.wiremock.stubbing.Scenario;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.agent.AIAgent;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;
import io.kestra.plugin.ai.langdock.LangdockModelFamily;
import io.kestra.plugin.ai.langdock.LangdockRegion;
import io.kestra.plugin.ai.tool.KestraTask;
import io.kestra.plugin.core.log.Log;

import jakarta.inject.Inject;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class LangdockTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance()
        .options(wireMockConfig().dynamicPort())
        .build();

    private String wireMockBaseUrl() {
        return "http://localhost:" + wireMock.getPort();
    }

    private Langdock.LangdockBuilder<?, ?> provider() {
        return Langdock.builder()
            .type(Langdock.class.getName())
            .apiKey(Property.ofValue("test-langdock-key"))
            .baseUrl(Property.ofValue(wireMockBaseUrl()));
    }

    // --- OpenAI family ---

    @Test
    void openAiFamily_shouldSendBearerAuthAndExcludeUnsupportedParams() throws Exception {
        wireMock.stubFor(
            post(urlPathEqualTo("/chat/completions"))
                .willReturn(okJson(openAiChatResponse("Hello John, nice to meet you!", null)))
        );

        var task = ChatCompletion.builder()
            .messages(Property.ofValue(List.of(ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build())))
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(42)).build())
            .provider(
                provider()
                    .modelFamily(Property.ofValue(LangdockModelFamily.OPENAI))
                    .modelName(Property.ofValue("gpt-5-mini"))
                    .build()
            )
            .build();

        var output = task.run(runContextFactory.of(Map.of()));

        assertThat(output.getTextOutput()).contains("John");

        wireMock.verify(
            postRequestedFor(urlPathEqualTo("/chat/completions"))
                .withHeader("Authorization", equalTo("Bearer test-langdock-key"))
                .withRequestBody(notMatching(".*\"service_tier\".*"))
                .withRequestBody(notMatching(".*\"parallel_tool_calls\".*"))
                .withRequestBody(notMatching(".*\"stream_options\".*"))
                .withRequestBody(notMatching(".*\"n\":\\d.*"))
        );
    }

    @Test
    void openAiFamily_toolCalls_shouldRoundTripThroughAIAgent() throws Exception {
        wireMock.stubFor(
            post(urlPathEqualTo("/chat/completions"))
                .inScenario("langdock-openai-tool")
                .whenScenarioStateIs(Scenario.STARTED)
                .willReturn(okJson(openAiToolCallResponse("kestra_task_log", "call_1", "{\"message\":\"Hello World!\"}")))
                .willSetStateTo("tool-executed")
        );
        wireMock.stubFor(
            post(urlPathEqualTo("/chat/completions"))
                .inScenario("langdock-openai-tool")
                .whenScenarioStateIs("tool-executed")
                .willReturn(okJson(openAiChatResponse("I logged the message successfully.", null)))
        );

        var agent = AIAgent.builder()
            .provider(
                provider()
                    .modelFamily(Property.ofValue(LangdockModelFamily.OPENAI))
                    .modelName(Property.ofValue("gpt-5-mini"))
                    .build()
            )
            .tools(List.of(KestraTask.builder().tasks(List.of(
                Log.builder().id("log").type(Log.class.getName()).message(Property.ofValue("...")).build()
            )).build()))
            .prompt(Property.ofValue("Log the message 'Hello World!'"))
            .build();

        var output = agent.run(runContextFactory.of(Map.of()));

        assertThat(output.getTextOutput()).contains("successfully");
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_task_log");

        wireMock.verify(2, postRequestedFor(urlPathEqualTo("/chat/completions"))
            .withHeader("Authorization", equalTo("Bearer test-langdock-key")));
    }

    @Test
    void openAiFamily_usRegion_shouldNotOverrideExplicitBaseUrl() throws Exception {
        // baseUrl always wins over region — verified here by hitting WireMock regardless of the region set.
        wireMock.stubFor(post(urlPathEqualTo("/chat/completions")).willReturn(okJson(openAiChatResponse("Hi John", null))));

        var task = ChatCompletion.builder()
            .messages(Property.ofValue(List.of(ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build())))
            .configuration(ChatConfiguration.empty())
            .provider(
                provider()
                    .modelFamily(Property.ofValue(LangdockModelFamily.OPENAI))
                    .region(Property.ofValue(LangdockRegion.US))
                    .modelName(Property.ofValue("gpt-5-mini"))
                    .build()
            )
            .build();

        var output = task.run(runContextFactory.of(Map.of()));

        assertThat(output.getTextOutput()).contains("Hi John");
        wireMock.verify(postRequestedFor(urlPathEqualTo("/chat/completions")));
    }

    @Test
    void topK_withOpenAiFamily_shouldBeRejected() throws Exception {
        var provider = provider().modelFamily(Property.ofValue(LangdockModelFamily.OPENAI)).modelName(Property.ofValue("gpt-5-mini")).build();
        var runContext = runContextFactory.of(Map.of());

        assertThatThrownBy(() -> provider.chatModel(runContext, ChatConfiguration.builder().topK(Property.ofValue(10)).build()))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("topK");
    }

    // --- Anthropic family ---

    @Test
    void anthropicFamily_shouldSendBearerAuthOnTopOfXApiKey() throws Exception {
        wireMock.stubFor(
            post(urlPathEqualTo("/messages"))
                .willReturn(okJson(anthropicTextResponse("Hello John, nice to meet you!")))
        );

        var task = ChatCompletion.builder()
            .messages(Property.ofValue(List.of(ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build())))
            .configuration(ChatConfiguration.builder().maxToken(Property.ofValue(256)).build())
            .provider(
                provider()
                    .modelFamily(Property.ofValue(LangdockModelFamily.ANTHROPIC))
                    .modelName(Property.ofValue("claude-sonnet-4-6-default"))
                    .build()
            )
            .build();

        var output = task.run(runContextFactory.of(Map.of()));

        assertThat(output.getTextOutput()).contains("John");

        wireMock.verify(
            postRequestedFor(urlPathEqualTo("/messages"))
                .withHeader("Authorization", equalTo("Bearer test-langdock-key"))
                .withHeader("x-api-key", equalTo("test-langdock-key"))
        );
    }

    @Test
    void anthropicFamily_toolUse_shouldRoundTripThroughAIAgent() throws Exception {
        wireMock.stubFor(
            post(urlPathEqualTo("/messages"))
                .inScenario("langdock-anthropic-tool")
                .whenScenarioStateIs(Scenario.STARTED)
                .willReturn(okJson(anthropicToolUseResponse("kestra_task_log", "toolu_1", Map.of("message", "Hello World!"))))
                .willSetStateTo("tool-executed")
        );
        wireMock.stubFor(
            post(urlPathEqualTo("/messages"))
                .inScenario("langdock-anthropic-tool")
                .whenScenarioStateIs("tool-executed")
                .willReturn(okJson(anthropicTextResponse("I logged the message successfully.")))
        );

        var agent = AIAgent.builder()
            .provider(
                provider()
                    .modelFamily(Property.ofValue(LangdockModelFamily.ANTHROPIC))
                    .modelName(Property.ofValue("claude-sonnet-4-6-default"))
                    .build()
            )
            .configuration(ChatConfiguration.builder().maxToken(Property.ofValue(256)).build())
            .tools(List.of(KestraTask.builder().tasks(List.of(
                Log.builder().id("log").type(Log.class.getName()).message(Property.ofValue("...")).build()
            )).build()))
            .prompt(Property.ofValue("Log the message 'Hello World!'"))
            .build();

        var output = agent.run(runContextFactory.of(Map.of()));

        assertThat(output.getTextOutput()).contains("successfully");
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_task_log");

        wireMock.verify(2, postRequestedFor(urlPathEqualTo("/messages"))
            .withHeader("Authorization", equalTo("Bearer test-langdock-key")));
    }

    @Test
    void seed_withAnthropicFamily_shouldBeRejected() throws Exception {
        var provider = provider().modelFamily(Property.ofValue(LangdockModelFamily.ANTHROPIC)).modelName(Property.ofValue("claude-sonnet-4-6-default")).build();
        var runContext = runContextFactory.of(Map.of());

        assertThatThrownBy(() -> provider.chatModel(runContext, ChatConfiguration.builder().seed(Property.ofValue(42)).build()))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("seed");
    }

    // --- Embeddings, image, misc ---

    @Test
    void embeddingModel_shouldAlwaysUseOpenAiRoute_evenWithAnthropicFamily() throws Exception {
        wireMock.stubFor(
            post(urlPathEqualTo("/embeddings"))
                .willReturn(okJson("""
                    {
                      "object": "list",
                      "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
                      "model": "text-embedding-ada-002",
                      "usage": {"prompt_tokens": 3, "total_tokens": 3}
                    }"""))
        );

        var provider = provider()
            .modelFamily(Property.ofValue(LangdockModelFamily.ANTHROPIC))
            .modelName(Property.ofValue("text-embedding-ada-002"))
            .build();
        var runContext = runContextFactory.of(Map.of());

        var embeddingModel = provider.embeddingModel(runContext);
        var result = embeddingModel.embed("hello world");

        assertThat(result.content().vector()).hasSize(3);
        wireMock.verify(postRequestedFor(urlPathEqualTo("/embeddings")));
    }

    @Test
    void imageModel_shouldThrowUnsupportedOperationException() throws Exception {
        var provider = provider().modelName(Property.ofValue("gpt-5-mini")).build();
        var runContext = runContextFactory.of(Map.of());

        assertThatThrownBy(() -> provider.imageModel(runContext))
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Langdock is currently not supported for image generation.");
    }

    // --- Integration tests: skipped unless a real Langdock key is available ---

    @Test
    @EnabledIfEnvironmentVariable(named = "LANGDOCK_API_KEY", matches = ".*")
    void integration_chatCompletionOnOpenAiRoute() throws Exception {
        var apiKey = System.getenv("LANGDOCK_API_KEY");
        var task = ChatCompletion.builder()
            .messages(Property.ofValue(List.of(ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build())))
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).build())
            .provider(
                Langdock.builder()
                    .type(Langdock.class.getName())
                    .apiKey(Property.ofValue(apiKey))
                    .modelFamily(Property.ofValue(LangdockModelFamily.OPENAI))
                    .modelName(Property.ofValue(Objects.requireNonNullElse(System.getenv("LANGDOCK_OPENAI_MODEL"), "gpt-5.4-mini")))
                    .build()
            )
            .build();

        var output = task.run(runContextFactory.of(Map.of()));
        assertThat(output.getTextOutput()).contains("John");
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "LANGDOCK_API_KEY", matches = ".*")
    void integration_agentWithToolOnAnthropicRoute() throws Exception {
        var apiKey = System.getenv("LANGDOCK_API_KEY");
        var agent = AIAgent.builder()
            .provider(
                Langdock.builder()
                    .type(Langdock.class.getName())
                    .apiKey(Property.ofValue(apiKey))
                    .modelFamily(Property.ofValue(LangdockModelFamily.ANTHROPIC))
                    .modelName(Property.ofValue("claude-sonnet-4-6-default"))
                    .build()
            )
            .configuration(ChatConfiguration.builder().maxToken(Property.ofValue(256)).build())
            .tools(List.of(KestraTask.builder().tasks(List.of(
                Log.builder().id("log").type(Log.class.getName()).message(Property.ofValue("...")).build()
            )).build()))
            .prompt(Property.ofValue("Log the message 'Hello World!' using the provided tool."))
            .build();

        var output = agent.run(runContextFactory.of(Map.of()));
        assertThat(output.getTextOutput()).isNotNull();
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "LANGDOCK_WORKSPACE_API_KEY", matches = ".*")
    void integration_embeddings() throws Exception {
        var apiKey = System.getenv("LANGDOCK_WORKSPACE_API_KEY");
        var provider = Langdock.builder()
            .type(Langdock.class.getName())
            .apiKey(Property.ofValue(apiKey))
            .modelName(Property.ofValue("text-embedding-ada-002"))
            .build();

        var embeddingModel = provider.embeddingModel(runContextFactory.of(Map.of()));
        var result = embeddingModel.embed("hello world");

        assertThat(result.content().vector()).isNotEmpty();
    }

    private String openAiChatResponse(String content, String toolCallsJson) {
        return """
            {
              "id": "chatcmpl-langdock-test",
              "object": "chat.completion",
              "model": "gpt-5-mini",
              "choices": [{
                "index": 0,
                "message": {
                  "role": "assistant",
                  "content": %s
                },
                "finish_reason": "stop"
              }],
              "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
            }""".formatted(content == null ? "null" : "\"" + content + "\"");
    }

    private String openAiToolCallResponse(String toolName, String toolCallId, String argumentsJson) {
        return """
            {
              "id": "chatcmpl-langdock-tool",
              "object": "chat.completion",
              "model": "gpt-5-mini",
              "choices": [{
                "index": 0,
                "message": {
                  "role": "assistant",
                  "content": null,
                  "tool_calls": [{
                    "id": "%s",
                    "type": "function",
                    "function": {"name": "%s", "arguments": %s}
                  }]
                },
                "finish_reason": "tool_calls"
              }],
              "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}
            }""".formatted(toolCallId, toolName, "\"" + argumentsJson.replace("\"", "\\\"") + "\"");
    }

    private String anthropicTextResponse(String text) {
        return """
            {
              "id": "msg_langdock_test",
              "type": "message",
              "role": "assistant",
              "model": "claude-sonnet-4-6-default",
              "content": [{"type": "text", "text": "%s"}],
              "stop_reason": "end_turn",
              "usage": {"input_tokens": 12, "output_tokens": 8}
            }""".formatted(text);
    }

    private String anthropicToolUseResponse(String toolName, String toolUseId, Map<String, Object> input) {
        var inputJson = input.entrySet().stream()
            .map(e -> "\"" + e.getKey() + "\":\"" + e.getValue() + "\"")
            .reduce((a, b) -> a + "," + b)
            .orElse("");
        return """
            {
              "id": "msg_langdock_tool",
              "type": "message",
              "role": "assistant",
              "model": "claude-sonnet-4-6-default",
              "content": [{"type": "tool_use", "id": "%s", "name": "%s", "input": {%s}}],
              "stop_reason": "tool_use",
              "usage": {"input_tokens": 20, "output_tokens": 10}
            }""".formatted(toolUseId, toolName, inputJson);
    }
}
