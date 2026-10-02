package io.kestra.plugin.ai.provider;

import java.util.List;

import org.junit.jupiter.api.Test;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.ai.domain.ModelProvider;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class FallbackTest {
    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void deserializesOrderedPolymorphicProvidersWithoutOuterModelName() throws Exception {
        Fallback fallback = JacksonMapper.ofYaml().readValue(
            """
                type: io.kestra.plugin.ai.provider.Fallback
                providers:
                  - type: io.kestra.plugin.ai.provider.OpenAI
                    modelName: gpt-5-mini
                    apiKey: openai-test-key
                  - type: io.kestra.plugin.ai.provider.GoogleGemini
                    modelName: gemini-2.5-flash
                    apiKey: gemini-test-key
                """,
            Fallback.class
        );

        RunContext runContext = runContextFactory.of();
        List<ModelProvider> providers = runContext.render(fallback.getProviders()).asList(ModelProvider.class);

        assertThat(fallback.getModelName()).isNull();
        assertThat(providers).hasSize(2);
        assertThat(providers.get(0)).isExactlyInstanceOf(OpenAI.class);
        assertThat(runContext.render(((OpenAI) providers.get(0)).getModelName()).as(String.class).orElseThrow()).isEqualTo("gpt-5-mini");
        assertThat(providers.get(1)).isExactlyInstanceOf(GoogleGemini.class);
        assertThat(runContext.render(((GoogleGemini) providers.get(1)).getModelName()).as(String.class).orElseThrow()).isEqualTo("gemini-2.5-flash");
    }
}
