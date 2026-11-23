package io.kestra.plugin.ai.provider;

import java.util.Map;

import org.junit.jupiter.api.Test;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.domain.ChatConfiguration;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class ReturnThinkingForwardingTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void openRouter_shouldForwardReturnThinking() throws Exception {
        assertThat(thinking(openRouter(ChatConfiguration.empty()))).isFalse();
        assertThat(thinking(openRouter(withReturnThinking(true)))).isTrue();
        assertThat(thinking(openRouter(withReturnThinking(false)))).isFalse();
    }

    @Test
    void gitHubModels_shouldForwardReturnThinking() throws Exception {
        assertThat(thinking(gitHubModels(ChatConfiguration.empty()))).isFalse();
        assertThat(thinking(gitHubModels(withReturnThinking(true)))).isTrue();
        assertThat(thinking(gitHubModels(withReturnThinking(false)))).isFalse();
    }

    // OpenAiChatModel keeps the flag in a private field and exposes no getter.
    private static boolean thinking(ChatModel model) throws Exception {
        var field = OpenAiChatModel.class.getDeclaredField("returnThinking");
        field.setAccessible(true);
        return field.getBoolean(model);
    }

    private static ChatConfiguration withReturnThinking(boolean value) {
        return ChatConfiguration.builder().returnThinking(Property.ofValue(value)).build();
    }

    // Building the model does not call any remote API.
    private ChatModel openRouter(ChatConfiguration configuration) throws Exception {
        var provider = OpenRouter.builder()
            .type(OpenRouter.class.getName())
            .modelName(Property.ofValue("openai/gpt-4o"))
            .apiKey(Property.ofValue("test-key"))
            .build();
        return provider.chatModel(runContextFactory.of(Map.of()), configuration);
    }

    private ChatModel gitHubModels(ChatConfiguration configuration) throws Exception {
        var provider = GitHubModels.builder()
            .type(GitHubModels.class.getName())
            .modelName(Property.ofValue("gpt-4o"))
            .gitHubToken(Property.ofValue("test-token"))
            .build();
        return provider.chatModel(runContextFactory.of(Map.of()), configuration);
    }
}
