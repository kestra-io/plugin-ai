package io.kestra.plugin.ai.provider;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.api.extension.RegisterExtension;

import com.github.tomakehurst.wiremock.http.Body;
import com.github.tomakehurst.wiremock.junit5.WireMockExtension;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.completion.ImageGeneration;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;

import jakarta.inject.Inject;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@KestraTest
class DockerModelTest {

    @Inject
    private RunContextFactory runContextFactory;

    @RegisterExtension
    static WireMockExtension dmrMock = WireMockExtension.newInstance()
        .options(wireMockConfig().dynamicPort())
        .build();

    // --- property defaults ---

    @Test
    void defaults_baseUrlShouldPointToLocalDockerModelRunner() throws Exception {
        var provider = DockerModel.builder()
            .type(DockerModel.class.getName())
            .modelName(Property.ofValue("ai/smollm2"))
            .build();

        var runContext = runContextFactory.of(Map.of());
        String resolved = runContext.render(provider.getBaseUrl()).as(String.class).orElseThrow();

        assertThat(resolved).isEqualTo("http://localhost:12434/engines/v1");
    }

    @Test
    void defaults_apiKeyShouldBeNotNeeded() throws Exception {
        var provider = DockerModel.builder()
            .type(DockerModel.class.getName())
            .modelName(Property.ofValue("ai/smollm2"))
            .build();

        var runContext = runContextFactory.of(Map.of());
        String resolved = provider.resolveApiKey(runContext);

        assertThat(resolved).isEqualTo("not-needed");
    }

    @Test
    void defaults_customApiKeyShouldBeRespected() throws Exception {
        var provider = DockerModel.builder()
            .type(DockerModel.class.getName())
            .modelName(Property.ofValue("ai/smollm2"))
            .apiKey(Property.ofValue("my-custom-key"))
            .build();

        var runContext = runContextFactory.of(Map.of());
        String resolved = provider.resolveApiKey(runContext);

        assertThat(resolved).isEqualTo("my-custom-key");
    }

    @Test
    void defaults_customBaseUrlShouldOverrideDefault() throws Exception {
        String customUrl = "http://172.17.0.1:12434/engines/v1";
        var provider = DockerModel.builder()
            .type(DockerModel.class.getName())
            .modelName(Property.ofValue("ai/smollm2"))
            .baseUrl(Property.ofValue(customUrl))
            .build();

        var runContext = runContextFactory.of(Map.of());
        String resolved = runContext.render(provider.getBaseUrl()).as(String.class).orElseThrow();

        assertThat(resolved).isEqualTo(customUrl);
    }

    // --- WireMock-based tests (no live DMR required) ---

    @Test
    void chatCompletion_shouldHitCorrectEndpointAndReturnResponse() throws Exception {
        dmrMock.stubFor(
            post(urlPathEqualTo("/engines/v1/chat/completions"))
                .willReturn(
                    aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "application/json")
                        .withResponseBody(Body.fromJsonBytes("""
                            {
                              "id": "chatcmpl-dmr-test",
                              "object": "chat.completion",
                              "model": "ai/smollm2",
                              "choices": [{
                                "index": 0,
                                "message": {
                                  "role": "assistant",
                                  "content": "Hello John, nice to meet you!"
                                },
                                "finish_reason": "stop"
                              }],
                              "usage": {
                                "prompt_tokens": 12,
                                "completion_tokens": 8,
                                "total_tokens": 20
                              }
                            }""".getBytes()))
                )
        );

        String wireMockBaseUrl = "http://localhost:" + dmrMock.getPort() + "/engines/v1";

        RunContext runContext = runContextFactory.of(
            Map.of(
                "messages", List.of(
                    ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
                )
            )
        );

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).build())
            .provider(
                DockerModel.builder()
                    .type(DockerModel.class.getName())
                    .modelName(Property.ofValue("ai/smollm2"))
                    .baseUrl(Property.ofValue(wireMockBaseUrl))
                    .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput()).isNotNull();
        assertThat(output.getTextOutput()).contains("John");

        dmrMock.verify(postRequestedFor(urlPathEqualTo("/engines/v1/chat/completions")));
    }

    @Test
    void imageModel_shouldRouteToDefaultDiffuserEndpoint() throws Exception {
        dmrMock.stubFor(
            post(urlPathEqualTo("/engines/diffusers/v1/images/generations"))
                .willReturn(
                    aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "application/json")
                        .withBody("{\"data\": [{\"url\": \"http://localhost/mock-image.png\"}]}")
                )
        );

        String wireMockBaseUrl = "http://localhost:" + dmrMock.getPort() + "/engines/v1";

        RunContext runContext = runContextFactory.of(Map.of());

        ImageGeneration task = ImageGeneration.builder()
            .prompt(Property.ofValue("A cat riding a bicycle"))
            .provider(
                DockerModel.builder()
                    .type(DockerModel.class.getName())
                    .modelName(Property.ofValue("ai/stable-diffusion"))
                    .baseUrl(Property.ofValue(wireMockBaseUrl))
                    .build()
            )
            .build();

        ImageGeneration.Output output = task.run(runContext);

        assertThat(output.getImageUrl()).isEqualTo("http://localhost/mock-image.png");

        dmrMock.verify(postRequestedFor(urlPathEqualTo("/engines/diffusers/v1/images/generations")));
    }

    @Test
    void imageModel_withCustomBaseUrl_shouldStillRewriteToDiffuserPath() throws Exception {
        dmrMock.stubFor(
            post(urlPathEqualTo("/engines/diffusers/v1/images/generations"))
                .willReturn(
                    aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "application/json")
                        .withBody("{\"data\": [{\"url\": \"http://localhost/mock-image-2.png\"}]}")
                )
        );

        // A different host string that still carries the standard /engines/v1 path segment.
        String customBaseUrl = "http://127.0.0.1:" + dmrMock.getPort() + "/engines/v1";

        RunContext runContext = runContextFactory.of(Map.of());

        ImageGeneration task = ImageGeneration.builder()
            .prompt(Property.ofValue("A dog surfing"))
            .provider(
                DockerModel.builder()
                    .type(DockerModel.class.getName())
                    .modelName(Property.ofValue("ai/stable-diffusion"))
                    .baseUrl(Property.ofValue(customBaseUrl))
                    .build()
            )
            .build();

        ImageGeneration.Output output = task.run(runContext);

        assertThat(output.getImageUrl()).isEqualTo("http://localhost/mock-image-2.png");

        dmrMock.verify(postRequestedFor(urlPathEqualTo("/engines/diffusers/v1/images/generations")));
    }

    @Test
    void imageModel_withNonMatchingBaseUrl_shouldFailFast() {
        var provider = DockerModel.builder()
            .type(DockerModel.class.getName())
            .modelName(Property.ofValue("ai/stable-diffusion"))
            .baseUrl(Property.ofValue("https://gateway.example.com/dmr"))
            .build();

        var runContext = runContextFactory.of(Map.of());

        assertThatThrownBy(() -> provider.imageModel(runContext))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("baseUrl")
            .hasMessageContaining("https://gateway.example.com/dmr");
    }

    // --- Live integration test (skipped unless DMR is reachable) ---

    @Test
    @EnabledIfEnvironmentVariable(named = "DOCKER_MODEL_RUNNER_AVAILABLE", matches = "true")
    void integration_chatCompletion_withLiveDmr() throws Exception {
        RunContext runContext = runContextFactory.of(
            Map.of(
                "messages", List.of(
                    ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
                )
            )
        );

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).build())
            .provider(
                DockerModel.builder()
                    .type(DockerModel.class.getName())
                    .modelName(Property.ofValue(System.getenv().getOrDefault("DOCKER_MODEL_RUNNER_MODEL", "ai/smollm2")))
                    .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput()).isNotNull();
        assertThat(output.getRequestDuration()).isNotNull();
    }
}
