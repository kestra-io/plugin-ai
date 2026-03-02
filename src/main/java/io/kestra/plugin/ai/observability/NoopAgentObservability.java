package io.kestra.plugin.ai.observability;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.service.Result;
import io.kestra.plugin.ai.domain.TokenUsage;

public final class NoopAgentObservability implements AgentObservability {
    public static final NoopAgentObservability INSTANCE = new NoopAgentObservability();

    private NoopAgentObservability() {
    }

    @Override
    public void onStart(String prompt, String systemMessage) {
        // no-op
    }

    @Override
    public void onToolArgumentsError(String toolName, String requestId, Throwable error) {
        // no-op
    }

    @Override
    public void onToolExecutionError(String toolName, String requestId, Throwable error) {
        // no-op
    }

    @Override
    public void onCompletion(Result<AiMessage> completion, TokenUsage tokenUsage) {
        // no-op
    }

    @Override
    public void onFailure(Throwable error) {
        // no-op
    }

    @Override
    public void close() {
        // no-op
    }
}
