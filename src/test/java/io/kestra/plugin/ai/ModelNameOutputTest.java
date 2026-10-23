package io.kestra.plugin.ai;

import java.util.Map;

import org.junit.jupiter.api.Test;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.domain.AIOutput;
import io.kestra.plugin.ai.provider.Fallback;
import io.kestra.plugin.ai.provider.OpenAI;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.ChatResponseMetadata;
import dev.langchain4j.service.Result;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class ModelNameOutputTest {
    @Inject
    private RunContextFactory runContextFactory;

    private OpenAI provider(String modelName) {
        return OpenAI.builder().apiKey(Property.ofValue("key")).modelName(Property.ofValue(modelName)).build();
    }

    @Test
    void reportedModelNameWinsOverConfiguredOne() throws Exception {
        var runContext = runContextFactory.of(Map.of());

        assertThat(AIUtils.modelName(runContext, provider("gpt-4o"), "gpt-4o-2024-08-06")).isEqualTo("gpt-4o-2024-08-06");
    }

    @Test
    void fallsBackToRenderedProviderModelNameWhenNothingIsReported() throws Exception {
        var runContext = runContextFactory.of(Map.of("model", "gpt-4o-mini"));
        var templated = OpenAI.builder().apiKey(Property.ofValue("key")).modelName(Property.ofExpression("{{ model }}")).build();

        assertThat(AIUtils.modelName(runContext, templated, null)).isEqualTo("gpt-4o-mini");
        assertThat(AIUtils.modelName(runContext, templated, " ")).isEqualTo("gpt-4o-mini");
        assertThat(AIUtils.modelName(runContext, templated, "unknown")).isEqualTo("gpt-4o-mini");
    }

    @Test
    void isNullWhenNeitherReportedNorConfigured() throws Exception {
        var runContext = runContextFactory.of(Map.of());

        assertThat(AIUtils.modelName(runContext, Fallback.builder().build(), null)).isNull();
    }

    @Test
    void fallbackProviderOutputCarriesTheReportedChildModelName() throws Exception {
        var runContext = runContextFactory.of(Map.of());
        var response = ChatResponse.builder()
            .aiMessage(AiMessage.from("hello"))
            .metadata(ChatResponseMetadata.builder().modelName("gemini-2.5-flash").build())
            .build();
        var result = Result.<AiMessage>builder().content(response.aiMessage()).finalResponse(response).build();

        var output = AIOutput.from(runContext, result, ResponseFormatType.TEXT, Fallback.builder().build());

        assertThat(output.getModelName()).isEqualTo("gemini-2.5-flash");
    }
}
