package io.kestra.plugin.ai.provider;

import java.util.List;

import org.junit.jupiter.api.Test;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.swagger.v3.oas.annotations.media.Schema;

import jakarta.inject.Inject;
import jakarta.validation.Validator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@KestraTest
class FallbackTest {
    @Inject
    private TestRunContextFactory runContextFactory;

    @Inject
    private Validator validator;

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

    @Test
    void validatesFallbackWithoutOuterModelName() {
        Fallback fallback = Fallback.builder()
            .type(Fallback.class.getName())
            .providers(Property.ofValue(List.of(
                OpenAI.builder()
                    .type(OpenAI.class.getName())
                    .modelName(Property.ofValue("gpt-5-mini"))
                    .apiKey(Property.ofValue("openai-test-key"))
                    .build(),
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .modelName(Property.ofValue("gemini-2.5-flash"))
                    .apiKey(Property.ofValue("gemini-test-key"))
                    .build()
            )))
            .build();

        assertThat(fallback.getModelName()).isNull();
        assertThat(validator.validate(fallback)).isEmpty();
    }

    @Test
    void validatesFallbackWithoutOuterConnectionProperties() {
        Fallback fallback = Fallback.builder()
            .type(Fallback.class.getName())
            .providers(Property.ofValue(List.of(
                OpenAI.builder()
                    .type(OpenAI.class.getName())
                    .modelName(Property.ofValue("gpt-5-mini"))
                    .apiKey(Property.ofValue("openai-test-key"))
                    .build()
            )))
            .build();

        assertThat(validator.validate(fallback)).isEmpty();
    }

    @Test
    void rejectsOuterBaseUrlBeforeConstructingChildChatModels() {
        assertOuterConnectionPropertyRejected(fallbackWithProvider().baseUrl(Property.ofValue("https://example.com")).build());
    }

    @Test
    void rejectsOuterClientPemBeforeConstructingChildChatModels() {
        assertOuterConnectionPropertyRejected(fallbackWithProvider().clientPem(Property.ofValue("client-pem")).build());
    }

    @Test
    void rejectsOuterCaPemBeforeConstructingChildChatModels() {
        assertOuterConnectionPropertyRejected(fallbackWithProvider().caPem(Property.ofValue("ca-pem")).build());
    }

    private void assertOuterConnectionPropertyRejected(Fallback fallback) {
        assertThatThrownBy(() -> fallback.chatModel(runContextFactory.of(), ChatConfiguration.empty()))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Fallback does not use outer `baseUrl`, `clientPem`, or `caPem`")
            .hasMessageContaining("appropriate nested provider instead");
    }

    private static Fallback.FallbackBuilder<?, ?> fallbackWithProvider() {
        return Fallback.builder()
            .type(Fallback.class.getName())
            .providers(Property.ofValue(List.of(
                OpenAI.builder()
                    .type(OpenAI.class.getName())
                    .modelName(Property.ofValue("gpt-5-mini"))
                    .apiKey(Property.ofValue("openai-test-key"))
                    .build()
            )));
    }

    @Test
    void validatesFallbackWithEmptyProviders() {
        Fallback fallback = Fallback.builder()
            .type(Fallback.class.getName())
            .providers(Property.ofValue(List.of()))
            .build();

        assertThat(validator.validate(fallback))
            .extracting(violation -> violation.getPropertyPath().toString())
            .contains("providers");
    }

    @Test
    void validatesModelNameForConcreteProvider() {
        OpenAI openAI = OpenAI.builder()
            .type(OpenAI.class.getName())
            .apiKey(Property.ofValue("openai-test-key"))
            .build();

        assertThat(validator.validate(openAI))
            .extracting(violation -> violation.getPropertyPath().toString())
            .contains("modelNameValid");
    }

    @Test
    void modelNameIsOptionalInPluginSchema() throws NoSuchFieldException {
        Schema modelNameSchema = ModelProvider.class.getDeclaredField("modelName").getAnnotation(Schema.class);

        assertThat(modelNameSchema.requiredMode()).isEqualTo(Schema.RequiredMode.NOT_REQUIRED);
    }
}
