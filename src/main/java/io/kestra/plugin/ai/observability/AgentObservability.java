package io.kestra.plugin.ai.observability;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.service.Result;
import io.kestra.plugin.ai.domain.TokenUsage;

public interface AgentObservability extends AutoCloseable {
    void onStart(String prompt, String systemMessage);

    void onToolArgumentsError(String toolName, String requestId, Throwable error);

    void onToolExecutionError(String toolName, String requestId, Throwable error);

    void onCompletion(Result<AiMessage> completion, TokenUsage tokenUsage);

    void onFailure(Throwable error);

    @Override
    void close();
}
