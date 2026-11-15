package io.kestra.plugin.ai.provider;

import java.util.Map;

import org.junit.jupiter.api.Test;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.domain.ChatConfiguration;

import dev.langchain4j.model.mistralai.MistralAiChatModel;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class MistralAITest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void chatModel_shouldLeaveReturnThinkingUnsetByDefault() throws Exception {
        assertThat(chatModel(ChatConfiguration.empty()).defaultRequestParameters().returnThinking()).isNull();
    }

    @Test
    void chatModel_shouldForwardReturnThinking() throws Exception {
        var enabled = ChatConfiguration.builder()
            .returnThinking(Property.ofValue(true))
            .build();
        var disabled = ChatConfiguration.builder()
            .returnThinking(Property.ofValue(false))
            .build();

        assertThat(chatModel(enabled).defaultRequestParameters().returnThinking()).isTrue();
        assertThat(chatModel(disabled).defaultRequestParameters().returnThinking()).isFalse();
    }

    // Building the model does not call the Mistral API.
    private MistralAiChatModel chatModel(ChatConfiguration configuration) throws Exception {
        var provider = MistralAI.builder()
            .type(MistralAI.class.getName())
            .modelName(Property.ofValue("mistral-small-latest"))
            .apiKey(Property.ofValue("test-key"))
            .build();

        return (MistralAiChatModel) provider.chatModel(runContextFactory.of(Map.of()), configuration);
    }
}
