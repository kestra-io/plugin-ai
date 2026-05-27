package io.kestra.plugin.ai.provider;

import java.util.Map;

import org.junit.jupiter.api.Test;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.domain.ChatConfiguration;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@KestraTest
class GoogleGeminiTest {
    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void resolveAuth_shouldAllowClientPemWithoutApiKey() throws Exception {
        var runContext = runContextFactory.of(
            Map.of("clientPem", "-----BEGIN CERTIFICATE-----\nplaceholder\n-----END CERTIFICATE-----")
        );

        var provider = GoogleGemini.builder()
            .type(GoogleGemini.class.getName())
            .modelName(Property.ofValue("gemini-2.5-flash"))
            .clientPem(Property.ofExpression("{{ clientPem }}"))
            .build();

        assertThat(provider.resolveAuth(runContext)).isNull();
    }

    @Test
    void resolveAuth_shouldRejectMissingAuthentication() throws Exception {
        var runContext = runContextFactory.of(Map.of());

        var provider = GoogleGemini.builder()
            .type(GoogleGemini.class.getName())
            .modelName(Property.ofValue("gemini-2.5-flash"))
            .build();

        assertThatThrownBy(() -> provider.resolveAuth(runContext))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage(
                "GoogleGemini requires either `apiKey` or `clientPem` (optionally with `caPem`) for certificate-based authentication."
            );
    }

    @Test
    void getThinkingConfig_shouldDefaultBudgetToZeroWhenNoConfigSet() throws Exception {
        // Reproduces issue #324: gemini-3.5-flash is a native thinking model — without an
        // explicit budget, the API attaches thought_signatures to function-call parts.
        // LangChain4j cannot propagate them in multi-turn tool calls, causing a 400 error.
        // Fix: default thinkingBudget to 0 so thinking is disabled unless the user opts in.
        var runContext = runContextFactory.of(Map.of());
        var provider = GoogleGemini.builder()
            .type(GoogleGemini.class.getName())
            .modelName(Property.ofValue("gemini-3.5-flash"))
            .apiKey(Property.ofValue("placeholder"))
            .build();
        var config = ChatConfiguration.empty();

        var thinkingConfig = provider.getThinkingConfig(config, runContext);

        assertThat(thinkingConfig.includeThoughts()).isFalse();
        assertThat(thinkingConfig.thinkingBudget()).isEqualTo(0);
    }

    @Test
    void getThinkingConfig_shouldRespectExplicitBudget() throws Exception {
        var runContext = runContextFactory.of(Map.of());
        var provider = GoogleGemini.builder()
            .type(GoogleGemini.class.getName())
            .modelName(Property.ofValue("gemini-3.5-flash"))
            .apiKey(Property.ofValue("placeholder"))
            .build();
        var config = ChatConfiguration.builder()
            .thinkingBudgetTokens(Property.ofValue(1024))
            .build();

        var thinkingConfig = provider.getThinkingConfig(config, runContext);

        assertThat(thinkingConfig.thinkingBudget()).isEqualTo(1024);
    }

    @Test
    void getThinkingConfig_shouldRespectThinkingEnabledTrue() throws Exception {
        var runContext = runContextFactory.of(Map.of());
        var provider = GoogleGemini.builder()
            .type(GoogleGemini.class.getName())
            .modelName(Property.ofValue("gemini-3.5-flash"))
            .apiKey(Property.ofValue("placeholder"))
            .build();
        var config = ChatConfiguration.builder()
            .thinkingEnabled(Property.ofValue(true))
            .build();

        var thinkingConfig = provider.getThinkingConfig(config, runContext);

        assertThat(thinkingConfig.includeThoughts()).isTrue();
        // When enabled=true but no budget set, budget stays null (let the model decide).
        assertThat(thinkingConfig.thinkingBudget()).isNull();
    }

    @Test
    void getThinkingConfig_shouldRespectExplicitThinkingEnabledWithBudget() throws Exception {
        var runContext = runContextFactory.of(Map.of());
        var provider = GoogleGemini.builder()
            .type(GoogleGemini.class.getName())
            .modelName(Property.ofValue("gemini-3.5-flash"))
            .apiKey(Property.ofValue("placeholder"))
            .build();
        var config = ChatConfiguration.builder()
            .thinkingEnabled(Property.ofValue(true))
            .thinkingBudgetTokens(Property.ofValue(512))
            .build();

        var thinkingConfig = provider.getThinkingConfig(config, runContext);

        assertThat(thinkingConfig.includeThoughts()).isTrue();
        assertThat(thinkingConfig.thinkingBudget()).isEqualTo(512);
    }
}
