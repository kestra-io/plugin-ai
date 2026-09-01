package io.kestra.plugin.ai.provider;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.domain.ChatConfiguration;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@ResourceLock("kestra-h2-flyway")
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
            .modelName(Property.ofValue("gemini-3.5-flash-lite"))
            .clientPem(Property.ofExpression("{{ clientPem }}"))
            .build();

        assertThat(provider.resolveAuth(runContext)).isNull();
    }

    @Test
    void resolveAuth_shouldRejectMissingAuthentication() throws Exception {
        var runContext = runContextFactory.of(Map.of());

        var provider = GoogleGemini.builder()
            .type(GoogleGemini.class.getName())
            .modelName(Property.ofValue("gemini-3.5-flash-lite"))
            .build();

        assertThatThrownBy(() -> provider.resolveAuth(runContext))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage(
                "GoogleGemini requires either `apiKey` or `clientPem` (optionally with `caPem`) for certificate-based authentication."
            );
    }

    @Test
    void getThinkingConfig_shouldDefaultBudgetToZeroOnGemini2WhenNoConfigSet() throws Exception {
        // Issue #324: thinking models attach thought_signatures to function-call parts.
        // Primary fix: returnThinking defaults to true (to capture the signature) and
        // sendThinking is always enabled (to re-attach it in follow-up requests), preventing
        // the 400 INVALID_ARGUMENT error from LangChain4j dropping the signature.
        // Belt-and-suspenders: on Gemini 2.x, thinkingBudget defaults to 0 to minimise thinking overhead.
        var runContext = runContextFactory.of(Map.of());
        var provider = GoogleGemini.builder()
            .type(GoogleGemini.class.getName())
            .modelName(Property.ofValue("gemini-2.5-flash"))
            .apiKey(Property.ofValue("placeholder"))
            .build();
        var config = ChatConfiguration.empty();

        var thinkingConfig = provider.getThinkingConfig(config, runContext);

        assertThat(thinkingConfig.includeThoughts()).isFalse();
        assertThat(thinkingConfig.thinkingBudget()).isEqualTo(0);
    }

    @Test
    void getThinkingConfig_shouldSendNoConfigOnGemini3AndLaterWhenNoConfigSet() throws Exception {
        // Gemini 3+ cannot have thinking turned off and rejects `thinkingBudget: 0` with
        // 400 INVALID_ARGUMENT, so no thinking configuration must be sent at all.
        var runContext = runContextFactory.of(Map.of());
        var config = ChatConfiguration.empty();

        for (var modelName : List.of("gemini-3.5-flash-lite", "gemini-3-pro-preview", "models/gemini-4-flash")) {
            var provider = GoogleGemini.builder()
                .type(GoogleGemini.class.getName())
                .modelName(Property.ofValue(modelName))
                .apiKey(Property.ofValue("placeholder"))
                .build();

            assertThat(provider.getThinkingConfig(config, runContext))
                .as("thinking config for %s", modelName)
                .isNull();
        }
    }

    @Test
    void getThinkingConfig_shouldRespectExplicitBudget() throws Exception {
        // On a Gemini 3 model an explicit budget is still sent: only the implicit 0 default is skipped.
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
