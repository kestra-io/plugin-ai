package io.kestra.plugin.ai.langdock;

import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.parallel.ResourceLock;

import com.github.tomakehurst.wiremock.junit5.WireMockExtension;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;

import jakarta.inject.Inject;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class ListModelsTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance()
        .options(wireMockConfig().dynamicPort())
        .build();

    private String wireMockBaseUrl() {
        return "http://localhost:" + wireMock.getPort();
    }

    private ListModels.ListModelsBuilder<?, ?> task() {
        return ListModels.builder()
            .id("list_models")
            .type(ListModels.class.getName())
            .apiKey(Property.ofValue("test-langdock-key"))
            .baseUrl(Property.ofValue(wireMockBaseUrl()));
    }

    @Test
    void openAiFormat_shouldParseIdCreatedAndOwnedBy() throws Exception {
        wireMock.stubFor(
            get(urlPathEqualTo("/models"))
                .willReturn(okJson("""
                    {
                      "object": "list",
                      "data": [
                        {"id": "gpt-5-mini", "object": "model", "created": 1700000000, "owned_by": "openai"},
                        {"id": "gpt-5", "object": "model", "created": 1700000001, "owned_by": "openai"}
                      ]
                    }"""))
        );

        var output = task().modelFamily(Property.ofValue(LangdockModelFamily.OPENAI)).build().run(runContextFactory.of(Map.of()));

        assertThat(output.getCount()).isEqualTo(2);
        assertThat(output.getModels()).extracting("id").containsExactly("gpt-5-mini", "gpt-5");
        assertThat(output.getModels().getFirst().getOwnedBy()).isEqualTo("openai");
        assertThat(output.getModels().getFirst().getCreatedAt()).isNotNull();

        wireMock.verify(getRequestedFor(urlPathEqualTo("/models"))
            .withHeader("Authorization", equalTo("Bearer test-langdock-key")));
    }

    @Test
    void anthropicFormat_shouldParseIdAndDisplayName() throws Exception {
        wireMock.stubFor(
            get(urlPathEqualTo("/models"))
                .willReturn(okJson("""
                    {
                      "data": [
                        {"id": "claude-sonnet-4-6-default", "type": "model", "display_name": "Claude Sonnet 4.6", "created_at": "2025-01-15T00:00:00Z"}
                      ],
                      "has_more": false
                    }"""))
        );

        var output = task().modelFamily(Property.ofValue(LangdockModelFamily.ANTHROPIC)).build().run(runContextFactory.of(Map.of()));

        assertThat(output.getCount()).isEqualTo(1);
        var model = output.getModels().getFirst();
        assertThat(model.getId()).isEqualTo("claude-sonnet-4-6-default");
        assertThat(model.getDisplayName()).isEqualTo("Claude Sonnet 4.6");
        assertThat(model.getCreatedAt()).isNotNull();
    }

    @Test
    void missingDataField_shouldFailWithClearError() throws Exception {
        wireMock.stubFor(get(urlPathEqualTo("/models")).willReturn(okJson("{\"unexpected\": true}")));

        var task = task().modelFamily(Property.ofValue(LangdockModelFamily.OPENAI)).build();

        assertThatThrownBy(() -> task.run(runContextFactory.of(Map.of())))
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("missing a 'data' array")
            .hasMessageContaining("OPENAI");
    }

    @Test
    void unauthorized_shouldFailWithActionableMessage() throws Exception {
        wireMock.stubFor(get(urlPathEqualTo("/models")).willReturn(aResponse().withStatus(401).withBody("{\"error\": \"invalid api key\"}")));

        var task = task().build();

        assertThatThrownBy(() -> task.run(runContextFactory.of(Map.of())))
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("401")
            .hasMessageContaining("Completion API scope");
    }

    @Test
    void rateLimited_shouldFailWithRetryHint() throws Exception {
        wireMock.stubFor(get(urlPathEqualTo("/models")).willReturn(aResponse().withStatus(429).withBody("{\"error\": \"rate limited\"}")));

        var task = task().build();

        assertThatThrownBy(() -> task.run(runContextFactory.of(Map.of())))
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("429")
            .hasMessageContaining("retry");
    }

    @Test
    void serverError_shouldFailWithStatusInMessage() throws Exception {
        wireMock.stubFor(get(urlPathEqualTo("/models")).willReturn(aResponse().withStatus(503).withBody("Service Unavailable")));

        var task = task().build();

        assertThatThrownBy(() -> task.run(runContextFactory.of(Map.of())))
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("503");
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "LANGDOCK_API_KEY", matches = ".*")
    void integration_listOpenAiModels() throws Exception {
        var apiKey = System.getenv("LANGDOCK_API_KEY");
        var output = ListModels.builder()
            .id("list_models")
            .type(ListModels.class.getName())
            .apiKey(Property.ofValue(apiKey))
            .modelFamily(Property.ofValue(LangdockModelFamily.OPENAI))
            .build()
            .run(runContextFactory.of(Map.of()));

        assertThat(output.getModels()).isNotEmpty();
    }
}
