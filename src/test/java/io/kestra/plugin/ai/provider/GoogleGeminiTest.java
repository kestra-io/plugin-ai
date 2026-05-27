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
        var runContext = runContextFactory.of(Map.of());
        var config = ChatConfiguration.empty();

        var thinkingConfig = GoogleGemini.getThinkingConfig(config, runContext);

        assertThat(thinkingConfig.includeThoughts()).isFalse();
        // Budget must be 0 (not null) so thinking models like gemini-3.5-flash do not attach
        // thought_signatures that LangChain4j cannot propagate in multi-turn tool calls.
        assertThat(thinkingConfig.thinkingBudget()).isEqualTo(0);
    }

    @Test
    void getThinkingConfig_shouldRespectExplicitBudget() throws Exception {
        var runContext = runContextFactory.of(Map.of());
        var config = ChatConfiguration.builder()
            .thinkingBudgetTokens(Property.ofValue(1024))
            .build();

        var thinkingConfig = GoogleGemini.getThinkingConfig(config, runContext);

        assertThat(thinkingConfig.thinkingBudget()).isEqualTo(1024);
    }

    @Test
    void getThinkingConfig_shouldRespectThinkingEnabledTrue() throws Exception {
        var runContext = runContextFactory.of(Map.of());
        var config = ChatConfiguration.builder()
            .thinkingEnabled(Property.ofValue(true))
            .build();

        var thinkingConfig = GoogleGemini.getThinkingConfig(config, runContext);

        assertThat(thinkingConfig.includeThoughts()).isTrue();
        // When enabled=true but no budget set, budget stays null (let the model decide).
        assertThat(thinkingConfig.thinkingBudget()).isNull();
    }

    @Test
    void getThinkingConfig_shouldRespectExplicitThinkingEnabledWithBudget() throws Exception {
        var runContext = runContextFactory.of(Map.of());
        var config = ChatConfiguration.builder()
            .thinkingEnabled(Property.ofValue(true))
            .thinkingBudgetTokens(Property.ofValue(512))
            .build();

        var thinkingConfig = GoogleGemini.getThinkingConfig(config, runContext);

        assertThat(thinkingConfig.includeThoughts()).isTrue();
        assertThat(thinkingConfig.thinkingBudget()).isEqualTo(512);
    }
}
