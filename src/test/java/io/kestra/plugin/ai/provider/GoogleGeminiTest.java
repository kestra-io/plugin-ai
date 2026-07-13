package io.kestra.plugin.ai.provider;

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
        // Issue #324: gemini-3.5-flash attaches thought_signatures to function-call parts.
        // Primary fix: returnThinking defaults to true (to capture the signature) and
        // sendThinking is always enabled (to re-attach it in follow-up requests), preventing
        // the 400 INVALID_ARGUMENT error from LangChain4j dropping the signature.
        // Belt-and-suspenders: thinkingBudget defaults to 0 to minimise thinking overhead.
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
