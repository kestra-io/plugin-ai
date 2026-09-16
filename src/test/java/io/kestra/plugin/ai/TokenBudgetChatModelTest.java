package io.kestra.plugin.ai;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.listener.ChatModelResponseContext;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.TokenUsage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TokenBudgetChatModelTest {
    private static final ChatRequest REQUEST = ChatRequest.builder()
        .messages(UserMessage.from("test"))
        .build();

    @Test
    void noBudgetReturnsOriginalModel() {
        var delegate = new StubChatModel(response(1, 1));

        assertThat(TokenBudgetChatModel.wrap(delegate, null)).isSameAs(delegate);
    }

    @Test
    void rejectsNonPositiveBudget() {
        var delegate = new StubChatModel(response(1, 1));

        assertThatThrownBy(() -> TokenBudgetChatModel.wrap(delegate, 0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("`maxCumulativeTokens` must be greater than 0.");
    }

    @Test
    void allowsResponsesUpToBudgetAndBlocksTheNextCall() {
        var delegate = new StubChatModel(response(10, 5), response(15, 10));
        var model = TokenBudgetChatModel.wrap(delegate, 40);

        assertThat(model.chat(REQUEST)).isNotNull();
        assertThat(model.chat(REQUEST)).isNotNull();
        assertThat(delegate.invocationCount).isEqualTo(2);

        assertThatThrownBy(() -> model.chat(REQUEST))
            .isInstanceOf(IllegalStateException.class)
            .hasMessage("Cumulative token budget exhausted: consumed 40 tokens, configured `maxCumulativeTokens` is 40.");
        assertThat(delegate.invocationCount).isEqualTo(2);
    }

    @Test
    void failsTheResponseThatExceedsBudgetAndBlocksFurtherCalls() {
        var delegate = new StubChatModel(response(10, 10), response(15, 10), response(1, 1));
        var model = TokenBudgetChatModel.wrap(delegate, 40);

        assertThat(model.chat(REQUEST)).isNotNull();
        assertThatThrownBy(() -> model.chat(REQUEST))
            .isInstanceOf(IllegalStateException.class)
            .hasMessage("Cumulative token budget exceeded: consumed 45 tokens, configured `maxCumulativeTokens` is 40.");
        assertThat(delegate.invocationCount).isEqualTo(2);

        assertThatThrownBy(() -> model.chat(REQUEST))
            .isInstanceOf(IllegalStateException.class)
            .hasMessage("Cumulative token budget exceeded: consumed 45 tokens, configured `maxCumulativeTokens` is 40.");
        assertThat(delegate.invocationCount).isEqualTo(2);
    }

    @Test
    void failsClosedWhenTokenUsageIsMissing() {
        var delegate = new StubChatModel(responseWithoutTokenUsage(), response(1, 1));
        var model = TokenBudgetChatModel.wrap(delegate, 40);

        assertThatThrownBy(() -> model.chat(REQUEST))
            .isInstanceOf(IllegalStateException.class)
            .hasMessage("Cannot enforce `maxCumulativeTokens` because the model response did not include total token usage.");
        assertThatThrownBy(() -> model.chat(REQUEST))
            .isInstanceOf(IllegalStateException.class)
            .hasMessage("Cannot enforce `maxCumulativeTokens` because the model response did not include total token usage.");
        assertThat(delegate.invocationCount).isEqualTo(1);
    }

    @Test
    void preservesModelCapabilities() {
        var delegate = new StubChatModel(response(1, 1));
        var model = TokenBudgetChatModel.wrap(delegate, 40);

        assertThat(model.supportedCapabilities()).containsExactly(Capability.RESPONSE_FORMAT_JSON_SCHEMA);
    }

    @Test
    void preservesModelListenersWithoutCallingThemTwice() {
        var responseCount = new AtomicInteger();
        var listener = new ChatModelListener() {
            @Override
            public void onResponse(ChatModelResponseContext responseContext) {
                responseCount.incrementAndGet();
            }
        };
        var delegate = new StubChatModel(listener, response(1, 1));
        var model = TokenBudgetChatModel.wrap(delegate, 40);

        assertThat(model.chat(REQUEST)).isNotNull();
        assertThat(responseCount).hasValue(1);
    }

    private static ChatResponse response(int inputTokens, int outputTokens) {
        return ChatResponse.builder()
            .aiMessage(AiMessage.from("response"))
            .tokenUsage(new TokenUsage(inputTokens, outputTokens))
            .build();
    }

    private static ChatResponse responseWithoutTokenUsage() {
        return ChatResponse.builder()
            .aiMessage(AiMessage.from("response"))
            .build();
    }

    private static final class StubChatModel implements ChatModel {
        private final Deque<ChatResponse> responses;
        private final List<ChatModelListener> listeners;
        private int invocationCount;

        private StubChatModel(ChatResponse... responses) {
            this(null, responses);
        }

        private StubChatModel(ChatModelListener listener, ChatResponse... responses) {
            this.responses = new ArrayDeque<>(Arrays.asList(responses));
            this.listeners = listener == null ? List.of() : List.of(listener);
        }

        @Override
        public ChatResponse doChat(ChatRequest request) {
            invocationCount++;
            return responses.removeFirst();
        }

        @Override
        public Set<Capability> supportedCapabilities() {
            return Set.of(Capability.RESPONSE_FORMAT_JSON_SCHEMA);
        }

        @Override
        public List<ChatModelListener> listeners() {
            return listeners;
        }
    }
}
