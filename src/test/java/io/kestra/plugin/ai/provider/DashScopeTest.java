package io.kestra.plugin.ai.provider;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.parallel.ResourceLock;

import com.github.tomakehurst.wiremock.junit5.WireMockExtension;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;

import jakarta.inject.Inject;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig;
import static org.assertj.core.api.Assertions.assertThat;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class DashScopeTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance()
        .options(wireMockConfig().dynamicPort())
        .build();

    @BeforeEach
    void stubGeneration() {
        wireMock.stubFor(
            post(urlPathMatching(".*/generation"))
                .willReturn(okJson("""
                    {
                      "request_id": "dashscope-test",
                      "output": {
                        "choices": [{
                          "finish_reason": "stop",
                          "message": {"role": "assistant", "content": "Hello John"}
                        }]
                      },
                      "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}
                    }"""))
        );
    }

    @Test
    void chatCompletion_shouldSendConfigurationMaxTokenAsMaxTokens() throws Exception {
        var task = ChatCompletion.builder()
            .messages(Property.ofValue(List.of(ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build())))
            .configuration(ChatConfiguration.builder().maxToken(Property.ofValue(42)).build())
            .provider(dashScope().build())
            .build();

        var output = task.run(runContextFactory.of(Map.of()));

        assertThat(output.getTextOutput()).contains("John");
        wireMock.verify(
            postRequestedFor(urlPathMatching(".*/generation"))
                .withRequestBody(matchingJsonPath("$.parameters.max_tokens", equalTo("42")))
        );
    }

    @Test
    void chatCompletion_shouldKeepProviderMaxTokensWhenConfigurationMaxTokenIsNotSet() throws Exception {
        var task = ChatCompletion.builder()
            .messages(Property.ofValue(List.of(ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build())))
            .provider(dashScope().maxTokens(Property.ofValue(64)).build())
            .build();

        task.run(runContextFactory.of(Map.of()));

        wireMock.verify(
            postRequestedFor(urlPathMatching(".*/generation"))
                .withRequestBody(matchingJsonPath("$.parameters.max_tokens", equalTo("64")))
        );
    }

    private DashScope.DashScopeBuilder<?, ?> dashScope() {
        return DashScope.builder()
            .type(DashScope.class.getName())
            .apiKey(Property.ofValue("test-dashscope-key"))
            .modelName(Property.ofValue("qwen-plus"))
            .baseUrl(Property.ofValue("http://localhost:" + wireMock.getPort() + "/api/v1"));
    }
}
