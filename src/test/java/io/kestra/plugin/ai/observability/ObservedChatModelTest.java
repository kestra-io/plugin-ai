package io.kestra.plugin.ai.observability;

import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.ChatRequestOptions;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import io.opentelemetry.api.OpenTelemetry;

import static org.assertj.core.api.Assertions.assertThat;

class ObservedChatModelTest {
    @Test
    void chatUsesOneArgumentOverrideWhenProviderDoesNotImplementDoChat() {
        var observed = observed(new ChatModel() {
            @Override
            public ChatResponse chat(ChatRequest chatRequest) {
                return ChatResponse.builder().aiMessage(AiMessage.from("legacy")).build();
            }
        });

        var response = observed.chat(request(), ChatRequestOptions.builder().addListenerAttribute("k", "v").build());

        assertThat(response.aiMessage().text()).isEqualTo("legacy");
    }

    @Test
    void chatKeepsRequestOptionsForProvidersThatImplementDoChat() {
        var observed = observed(new ChatModel() {
            @Override
            public ChatResponse chat(ChatRequest chatRequest, ChatRequestOptions options) {
                assertThat(options.listenerAttributes()).containsEntry("k", "v");
                return ChatModel.super.chat(chatRequest, options);
            }

            @Override
            public ChatResponse doChat(ChatRequest chatRequest) {
                return ChatResponse.builder().aiMessage(AiMessage.from("modern")).build();
            }
        });

        var response = observed.chat(request(), ChatRequestOptions.builder().addListenerAttribute("k", "v").build());

        assertThat(response.aiMessage().text()).isEqualTo("modern");
    }

    private static ObservedChatModel observed(ChatModel delegate) {
        return new ObservedChatModel(delegate, new TelemetrySession(
            OpenTelemetry.noop().getTracer("test"),
            LoggerFactory.getLogger(ObservedChatModelTest.class),
            "localai",
            "model",
            null,
            null,
            false,
            null,
            null
        ));
    }

    private static ChatRequest request() {
        return ChatRequest.builder().messages(UserMessage.from("hi")).modelName("gpt-test").build();
    }
}
