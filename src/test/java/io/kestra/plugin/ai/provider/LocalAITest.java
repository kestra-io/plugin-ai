package io.kestra.plugin.ai.provider;

import java.time.Duration;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.parallel.ResourceLock;

import com.github.tomakehurst.wiremock.junit5.WireMockExtension;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.domain.ChatConfiguration;

import jakarta.inject.Inject;

import static com.github.tomakehurst.wiremock.client.WireMock.okJson;
import static com.github.tomakehurst.wiremock.client.WireMock.post;
import static com.github.tomakehurst.wiremock.client.WireMock.urlPathMatching;
import static com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class LocalAITest {

    private static final String RESPONSE = """
        {
          "id": "chatcmpl-1",
          "object": "chat.completion",
          "model": "test-model",
          "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello John"}}],
          "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        }""";

    @Inject
    private TestRunContextFactory runContextFactory;

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance()
        .options(wireMockConfig().dynamicPort())
        .build();

    @Test
    void chatModel_shouldApplyTimeout() throws Exception {
        wireMock.stubFor(post(urlPathMatching(".*/chat/completions")).willReturn(okJson(RESPONSE).withFixedDelay(3000)));

        var model = localAi().chatModel(runContextFactory.of(Map.of()), ChatConfiguration.builder().build(), Duration.ofMillis(500));

        assertThatThrownBy(() -> model.chat("Hello"))
            .hasStackTraceContaining("timed out");
    }

    @Test
    void chatModel_shouldReturnResponse() throws Exception {
        wireMock.stubFor(post(urlPathMatching(".*/chat/completions")).willReturn(okJson(RESPONSE)));

        var model = localAi().chatModel(runContextFactory.of(Map.of()), ChatConfiguration.builder().build(), Duration.ofSeconds(5));

        assertThat(model.chat("Hello")).isEqualTo("Hello John");
    }

    private LocalAI localAi() {
        return LocalAI.builder()
            .type(LocalAI.class.getName())
            .modelName(Property.ofValue("test-model"))
            .baseUrl(Property.ofValue("http://localhost:" + wireMock.getPort() + "/v1"))
            .build();
    }
}
