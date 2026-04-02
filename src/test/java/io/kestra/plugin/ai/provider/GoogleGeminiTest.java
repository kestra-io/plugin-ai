package io.kestra.plugin.ai.provider;

import java.util.Map;

import org.junit.jupiter.api.Test;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;

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
}
