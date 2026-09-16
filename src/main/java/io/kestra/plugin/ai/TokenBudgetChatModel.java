package io.kestra.plugin.ai;

import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;

import dev.langchain4j.model.ModelProvider;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.ChatRequestOptions;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;

public final class TokenBudgetChatModel implements ChatModel {
    private final ChatModel delegate;
    private final long maxCumulativeTokens;

    private long consumedTokens;
    private String failureMessage;

    private TokenBudgetChatModel(ChatModel delegate, int maxCumulativeTokens) {
        this.delegate = Objects.requireNonNull(delegate);
        this.maxCumulativeTokens = maxCumulativeTokens;
    }

    public static ChatModel wrap(
        ChatModel delegate,
        RunContext runContext,
        ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        Integer maxCumulativeTokens = runContext.render(configuration.getMaxCumulativeTokens())
            .as(Integer.class)
            .orElse(null);

        return wrap(delegate, maxCumulativeTokens);
    }

    static ChatModel wrap(ChatModel delegate, Integer maxCumulativeTokens) {
        if (maxCumulativeTokens == null) {
            return delegate;
        }

        if (maxCumulativeTokens <= 0) {
            throw new IllegalArgumentException("`maxCumulativeTokens` must be greater than 0.");
        }

        return new TokenBudgetChatModel(delegate, maxCumulativeTokens);
    }

    @Override
    public ChatResponse chat(ChatRequest request, ChatRequestOptions options) {
        return invoke(() -> delegate.chat(request, options));
    }

    @Override
    public ChatResponse doChat(ChatRequest request) {
        return invoke(() -> delegate.doChat(request));
    }

    @Override
    public ChatRequestParameters defaultRequestParameters() {
        return delegate.defaultRequestParameters();
    }

    @Override
    public List<ChatModelListener> listeners() {
        return delegate.listeners();
    }

    @Override
    public ModelProvider provider() {
        return delegate.provider();
    }

    @Override
    public Set<Capability> supportedCapabilities() {
        return delegate.supportedCapabilities();
    }

    private synchronized ChatResponse invoke(Supplier<ChatResponse> invocation) {
        ensureBudgetAvailable();

        ChatResponse response = invocation.get();
        var tokenUsage = response.tokenUsage();
        if (tokenUsage == null || tokenUsage.totalTokenCount() == null) {
            failureMessage = "Cannot enforce `maxCumulativeTokens` because the model response did not include total token usage.";
            throw new IllegalStateException(failureMessage);
        }

        consumedTokens += tokenUsage.totalTokenCount();
        if (consumedTokens > maxCumulativeTokens) {
            throw budgetException("exceeded");
        }

        return response;
    }

    private void ensureBudgetAvailable() {
        if (failureMessage != null) {
            throw new IllegalStateException(failureMessage);
        }

        if (consumedTokens >= maxCumulativeTokens) {
            throw budgetException(consumedTokens == maxCumulativeTokens ? "exhausted" : "exceeded");
        }
    }

    private IllegalStateException budgetException(String state) {
        return new IllegalStateException(
            "Cumulative token budget %s: consumed %d tokens, configured `maxCumulativeTokens` is %d."
                .formatted(state, consumedTokens, maxCumulativeTokens)
        );
    }
}
