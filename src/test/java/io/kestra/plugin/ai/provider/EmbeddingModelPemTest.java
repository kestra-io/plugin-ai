package io.kestra.plugin.ai.provider;

import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.domain.ModelProvider;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class EmbeddingModelPemTest {

    // Not a usable certificate: building the mTLS HTTP client from it must fail,
    // which proves that the embedding model reads `clientPem` instead of ignoring it.
    private static final String INVALID_PEM = "-----BEGIN CERTIFICATE-----\nplaceholder\n-----END CERTIFICATE-----";

    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void openAiCompliant_embeddingModel_shouldUseClientPem() {
        assertUsesClientPem(
            OpenAI.builder()
                .type(OpenAI.class.getName())
                .modelName(Property.ofValue("text-embedding-3-small"))
                .apiKey(Property.ofValue("placeholder"))
                .baseUrl(Property.ofValue("https://localhost/v1"))
                .clientPem(Property.ofValue(INVALID_PEM))
                .build()
        );
    }

    @Test
    void ollama_embeddingModel_shouldUseClientPem() {
        assertUsesClientPem(
            Ollama.builder()
                .type(Ollama.class.getName())
                .modelName(Property.ofValue("nomic-embed-text"))
                .endpoint(Property.ofValue("https://localhost"))
                .clientPem(Property.ofValue(INVALID_PEM))
                .build()
        );
    }

    @Test
    void mistralAi_embeddingModel_shouldUseClientPem() {
        assertUsesClientPem(
            MistralAI.builder()
                .type(MistralAI.class.getName())
                .modelName(Property.ofValue("mistral-embed"))
                .apiKey(Property.ofValue("placeholder"))
                .baseUrl(Property.ofValue("https://localhost/v1"))
                .clientPem(Property.ofValue(INVALID_PEM))
                .build()
        );
    }

    @Test
    void openRouter_embeddingModel_shouldUseClientPem() {
        assertUsesClientPem(
            OpenRouter.builder()
                .type(OpenRouter.class.getName())
                .modelName(Property.ofValue("openai/text-embedding-3-small"))
                .apiKey(Property.ofValue("placeholder"))
                .baseUrl(Property.ofValue("https://localhost/v1"))
                .clientPem(Property.ofValue(INVALID_PEM))
                .build()
        );
    }

    private void assertUsesClientPem(ModelProvider provider) {
        var runContext = runContextFactory.of(Map.of());

        assertThatThrownBy(() -> provider.embeddingModel(runContext))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("Exception while trying to setup AI Service certificates");
    }
}
