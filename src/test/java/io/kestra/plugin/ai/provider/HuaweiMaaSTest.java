
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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class HuaweiMaaSTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance()
            .options(wireMockConfig().dynamicPort())
            .build();

    private HuaweiMaaS.HuaweiMaaSBuilder<?, ?> provider() {
        return HuaweiMaaS.builder()
                .type(HuaweiMaaS.class.getName())
                .apiKey(Property.ofValue("test-huawei-key"))
                .modelName(Property.ofValue("glm-5.2"))
                .baseUrl(Property.ofValue("http://localhost:" + wireMock.getPort() + "/openai/v1"));
    }

    private ChatCompletion task() {
        return ChatCompletion.builder()
                .messages(Property.ofValue(List.of(
                        ChatMessage.builder()
                                .type(ChatMessageType.USER)
                                .content("Hello")
                                .build())))
                .provider(provider().build())
                .build();
    }

    @Test
    void chatCompletion_shouldSendBearerAuthAndReturnResponse() throws Exception {
        wireMock.stubFor(
                post(urlPathEqualTo("/openai/v1/chat/completions"))
                        .willReturn(okJson("""
                                {
                                  "id": "chatcmpl-test",
                                  "object": "chat.completion",
                                  "created": 1700000000,
                                  "model": "glm-5.2",
                                  "choices": [{
                                    "index": 0,
                                    "message": {
                                      "role": "assistant",
                                      "content": "Hello from Huawei MaaS"
                                    },
                                    "finish_reason": "stop"
                                  }],
                                  "usage": {
                                    "prompt_tokens": 5,
                                    "completion_tokens": 5,
                                    "total_tokens": 10
                                  }
                                }
                                """)));

        var output = task().run(runContextFactory.of(Map.of()));

        assertThat(output.getTextOutput()).contains("Hello from Huawei MaaS");

        wireMock.verify(
                postRequestedFor(urlPathEqualTo("/openai/v1/chat/completions"))
                        .withHeader("Authorization", equalTo("Bearer test-huawei-key"))
                        .withRequestBody(matchingJsonPath("$.model", equalTo("glm-5.2"))));
    }

    @Test
    void chatCompletion_shouldFailOnAuthenticationError() {
        wireMock.stubFor(
                post(urlPathEqualTo("/openai/v1/chat/completions"))
                        .willReturn(
                                aResponse()
                                        .withStatus(401)
                                        .withHeader("Content-Type", "application/json")
                                        .withBody(
                                                """
                                                        {"error": {"message": "Invalid API key", "type": "authentication_error"}}
                                                        """)));

        assertThatThrownBy(() -> task().run(runContextFactory.of(Map.of())))
                .isInstanceOf(Exception.class);

        wireMock.verify(
                postRequestedFor(urlPathEqualTo("/openai/v1/chat/completions"))
                        .withHeader("Authorization", equalTo("Bearer test-huawei-key")));
    }

    @Test
    void imageGeneration_shouldFailWithClearMessage() {
        var provider = provider().build();
        var runContext = runContextFactory.of(Map.of());

        assertThatThrownBy(() -> provider.imageModel(runContext))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("does not currently support image generation");
    }

    @Test
    void embeddings_shouldFailWithClearMessage() {
        var provider = provider().build();
        var runContext = runContextFactory.of(Map.of());

        assertThatThrownBy(() -> provider.embeddingModel(runContext))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("does not currently support embeddings");
    }
}
