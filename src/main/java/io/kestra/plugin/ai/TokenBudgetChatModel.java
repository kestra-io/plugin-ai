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
    private final RunContext runContext;
    private final long maxCumulativeTokens;

    private long consumedInputTokens;
    private long consumedOutputTokens;
    private long consumedTokens;
    private boolean tokenUsageRecorded;
    private boolean failureMetricsSent;
    private String failureMessage;

    private TokenBudgetChatModel(ChatModel delegate, RunContext runContext, int maxCumulativeTokens) {
        this.delegate = Objects.requireNonNull(delegate);
        this.runContext = runContext;
        this.maxCumulativeTokens = maxCumulativeTokens;
    }

    public static ChatModel wrap(
        ChatModel delegate,
        RunContext runContext,
        ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        var rMaxCumulativeTokens = runContext.render(configuration.getMaxCumulativeTokens())
            .as(Integer.class)
            .orElse(null);

        return wrap(delegate, rMaxCumulativeTokens, runContext);
    }

    static ChatModel wrap(ChatModel delegate, Integer maxCumulativeTokens) {
        return wrap(delegate, maxCumulativeTokens, null);
    }

    static ChatModel wrap(ChatModel delegate, Integer maxCumulativeTokens, RunContext runContext) {
        if (maxCumulativeTokens == null) {
            return delegate;
        }

        if (maxCumulativeTokens <= 0) {
            throw new IllegalArgumentException("`maxCumulativeTokens` must be greater than 0.");
        }

        return new TokenBudgetChatModel(delegate, runContext, maxCumulativeTokens);
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

    // Model calls within one AI service are sequential, so serializing them has no throughput cost and keeps the counter consistent.
    private synchronized ChatResponse invoke(Supplier<ChatResponse> invocation) {
        ensureBudgetAvailable();

        var response = invocation.get();
        var tokenUsage = response.tokenUsage();
        if (tokenUsage == null || tokenUsage.totalTokenCount() == null) {
            failureMessage = "Cannot enforce `maxCumulativeTokens` because the model response did not include total token usage. " +
                "Remove `maxCumulativeTokens` from the configuration, or use a provider/model that reports token usage.";
            sendFailureMetrics();
            throw new IllegalStateException(failureMessage);
        }

        consumedInputTokens += Objects.requireNonNullElse(tokenUsage.inputTokenCount(), 0);
        consumedOutputTokens += Objects.requireNonNullElse(tokenUsage.outputTokenCount(), 0);
        consumedTokens += tokenUsage.totalTokenCount();
        tokenUsageRecorded = true;
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
        sendFailureMetrics();
        return new IllegalStateException(
            ("Cumulative token budget %s: consumed %d tokens, configured `maxCumulativeTokens` is %d. " +
                "Increase `maxCumulativeTokens`, reduce the prompt or tool outputs, or limit `maxSequentialToolsInvocations`.")
                .formatted(state, consumedTokens, maxCumulativeTokens)
        );
    }

    private void sendFailureMetrics() {
        if (runContext != null && tokenUsageRecorded && !failureMetricsSent) {
            AIUtils.sendMetrics(runContext, consumedInputTokens, consumedOutputTokens, consumedTokens);
            failureMetricsSent = true;
        }
    }
}
