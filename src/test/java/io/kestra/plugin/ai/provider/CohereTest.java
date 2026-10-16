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
class CohereTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance()
        .options(wireMockConfig().dynamicPort())
        .build();

    @Test
    void baseUrl_shouldDefaultToCohereCompatibilityApi() throws Exception {
        Cohere cohere = Cohere.builder()
            .type(Cohere.class.getName())
            .apiKey(Property.ofValue("test-cohere-key"))
            .modelName(Property.ofValue("command-a-03-2025"))
            .build();

        RunContext runContext = runContextFactory.of(Map.of());

        assertThat(runContext.render(cohere.getBaseUrl()).as(String.class).orElseThrow())
            .isEqualTo("https://api.cohere.ai/compatibility/v1");
    }

    @Test
    void chatCompletion_shouldSendBearerAuthToChatCompletions() throws Exception {
        wireMock.stubFor(
            post(urlPathEqualTo("/chat/completions"))
                .willReturn(okJson("""
                    {
                      "id": "chatcmpl-cohere-test",
                      "object": "chat.completion",
                      "model": "command-a-03-2025",
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
                Cohere.builder()
                    .type(Cohere.class.getName())
                    .apiKey(Property.ofValue("test-cohere-key"))
                    .modelName(Property.ofValue("command-a-03-2025"))
                    .baseUrl(Property.ofValue("http://localhost:" + wireMock.getPort()))
                    .build()
            )
            .build();

        var output = task.run(runContextFactory.of(Map.of()));

        assertThat(output.getTextOutput()).contains("John");

        wireMock.verify(
            postRequestedFor(urlPathEqualTo("/chat/completions"))
                .withHeader("Authorization", equalTo("Bearer test-cohere-key"))
                .withRequestBody(matchingJsonPath("$.model", equalTo("command-a-03-2025")))
        );
    }
}
