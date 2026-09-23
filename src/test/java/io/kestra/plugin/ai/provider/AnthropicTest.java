package io.kestra.plugin.ai.provider;

import java.util.Map;

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

/**
 * Regression test for the {@code customHeaders} extension point added for Langdock support
 * (see {@link Langdock}): the default hook must add no header, so existing Anthropic users see no
 * behaviour change.
 */
@ResourceLock("kestra-h2-flyway")
@KestraTest
class AnthropicTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance()
        .options(wireMockConfig().dynamicPort())
        .build();

    @Test
    void customHeaders_defaultsToEmpty_soNoAuthorizationHeaderIsSent() throws Exception {
        var provider = Anthropic.builder()
            .type(Anthropic.class.getName())
            .apiKey(Property.ofValue("test-anthropic-key"))
            .modelName(Property.ofValue("claude-3-haiku-20240307"))
            .build();

        assertThat(provider.customHeaders(runContextFactory.of(Map.of()))).isEmpty();
    }

    @Test
    void chatModel_shouldOnlySendXApiKeyHeader_noAuthorizationHeader() throws Exception {
        wireMock.stubFor(
            post(urlPathEqualTo("/v1/messages"))
                .willReturn(okJson("""
                    {
                      "id": "msg_anthropic_test",
                      "type": "message",
                      "role": "assistant",
                      "model": "claude-3-haiku-20240307",
                      "content": [{"type": "text", "text": "Hello John"}],
                      "stop_reason": "end_turn",
                      "usage": {"input_tokens": 10, "output_tokens": 5}
                    }"""))
        );

        var task = ChatCompletion.builder()
            .messages(Property.ofValue(java.util.List.of(ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build())))
            .configuration(ChatConfiguration.builder().maxToken(Property.ofValue(256)).build())
            .provider(
                Anthropic.builder()
                    .type(Anthropic.class.getName())
                    .apiKey(Property.ofValue("test-anthropic-key"))
                    .modelName(Property.ofValue("claude-3-haiku-20240307"))
                    .baseUrl(Property.ofValue("http://localhost:" + wireMock.getPort() + "/v1/"))
                    .build()
            )
            .build();

        var output = task.run(runContextFactory.of(Map.of()));

        assertThat(output.getTextOutput()).contains("John");
        wireMock.verify(
            postRequestedFor(urlPathEqualTo("/v1/messages"))
                .withHeader("x-api-key", equalTo("test-anthropic-key"))
                .withoutHeader("Authorization")
        );
    }
}
