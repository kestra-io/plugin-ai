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
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;

import jakarta.inject.Inject;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig;
import static org.assertj.core.api.Assertions.assertThat;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class CheaperInferenceTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance()
        .options(wireMockConfig().dynamicPort())
        .build();

    @Test
    void baseUrl_shouldDefaultToCheaperInferenceApi() throws Exception {
        CheaperInference cheaperInference = CheaperInference.builder()
            .type(CheaperInference.class.getName())
            .apiKey(Property.ofValue("test-cheaper-inference-key"))
            .modelName(Property.ofValue("gpt-5.4-mini"))
            .build();

        RunContext runContext = runContextFactory.of(Map.of());

        assertThat(runContext.render(cheaperInference.getBaseUrl()).as(String.class).orElseThrow())
            .isEqualTo("https://api.cheaperinference.com/v1");
    }

    @Test
    void chatCompletion_shouldSendBearerAuthToChatCompletions() throws Exception {
        wireMock.stubFor(
            post(urlPathEqualTo("/chat/completions"))
                .willReturn(okJson("""
                    {
                      "id": "chatcmpl-cheaper-inference-test",
                      "object": "chat.completion",
                      "model": "gpt-5.4-mini",
                      "choices": [{
                        "index": 0,
                        "message": {
                          "role": "assistant",
                          "content": "Hello John, nice to meet you!"
                        },
                        "finish_reason": "stop"
                      }],
                      "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
                    }"""))
        );

        var task = ChatCompletion.builder()
            .messages(Property.ofValue(List.of(ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build())))
            .provider(
                CheaperInference.builder()
                    .type(CheaperInference.class.getName())
                    .apiKey(Property.ofValue("test-cheaper-inference-key"))
                    .modelName(Property.ofValue("gpt-5.4-mini"))
                    .baseUrl(Property.ofValue("http://localhost:" + wireMock.getPort()))
                    .build()
            )
            .build();

        var output = task.run(runContextFactory.of(Map.of()));

        assertThat(output.getTextOutput()).contains("John");

        wireMock.verify(
            postRequestedFor(urlPathEqualTo("/chat/completions"))
                .withHeader("Authorization", equalTo("Bearer test-cheaper-inference-key"))
                .withRequestBody(matchingJsonPath("$.model", equalTo("gpt-5.4-mini")))
        );
    }
}
