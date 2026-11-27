package io.kestra.plugin.ai.provider;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.parallel.ResourceLock;

import com.github.tomakehurst.wiremock.junit5.WireMockExtension;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;

import jakarta.inject.Inject;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig;
import static org.assertj.core.api.Assertions.assertThat;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class XAITest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance()
        .options(wireMockConfig().dynamicPort())
        .build();

    @Test
    void baseUrl_shouldDefaultToXaiApi() throws Exception {
        XAI xai = XAI.builder()
            .type(XAI.class.getName())
            .apiKey(Property.ofValue("test-xai-key"))
            .modelName(Property.ofValue("grok-3"))
            .build();

        var runContext = runContextFactory.of(Map.of());

        assertThat(runContext.render(xai.getBaseUrl()).as(String.class).orElseThrow())
            .isEqualTo("https://api.x.ai/v1");
    }

    @Test
    void chatCompletion_shouldSendBearerAuthToChatCompletions() throws Exception {
        wireMock.stubFor(
            post(urlPathEqualTo("/chat/completions"))
                .willReturn(okJson("""
                    {
                      "id": "chatcmpl-xai-test",
                      "object": "chat.completion",
                      "model": "grok-3",
                      "choices": [{
                        "index": 0,
                        "message": {
                          "role": "assistant",
                          "content": "Hello from Grok!"
                        },
                        "finish_reason": "stop"
                      }],
                      "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15
                      }
                    }"""))
        );

        var task = ChatCompletion.builder()
            .messages(Property.ofValue(List.of(
                ChatMessage.builder()
                    .type(ChatMessageType.USER)
                    .content("Hello")
                    .build()
            )))
            .provider(
                XAI.builder()
                    .type(XAI.class.getName())
                    .apiKey(Property.ofValue("test-xai-key"))
                    .modelName(Property.ofValue("grok-3"))
                    .baseUrl(Property.ofValue("http://localhost:" + wireMock.getPort()))
                    .build()
            )
            .build();

        var output = task.run(runContextFactory.of(Map.of()));

        assertThat(output.getTextOutput()).contains("Hello from Grok!");

        wireMock.verify(
            postRequestedFor(urlPathEqualTo("/chat/completions"))
                .withHeader("Authorization", equalTo("Bearer test-xai-key"))
                .withRequestBody(matchingJsonPath("$.model", equalTo("grok-3")))
        );
    }
}