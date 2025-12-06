package io.kestra.plugin.ai;

import com.fasterxml.jackson.core.JsonProcessingException;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.service.tool.ToolExecutor;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.ai.domain.TokenUsage;
import io.kestra.plugin.ai.domain.ToolProvider;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static io.kestra.core.utils.Rethrow.throwConsumer;

public final class AIUtils {

    private AIUtils() {
        // utility class pattern
    }

    public static Map<ToolSpecification, ToolExecutor> buildTools(RunContext runContext, Map<String, Object> additionalVariables, List<ToolProvider> toolProviders) throws Exception {
        if (toolProviders.isEmpty()) {
            return Collections.emptyMap();
        }

        Map<ToolSpecification, ToolExecutor> tools = new HashMap<>();
        toolProviders.forEach(throwConsumer(provider -> tools.putAll(provider.tool(runContext, additionalVariables))));
        return tools;
    }

    public static void sendMetrics(RunContext runContext, TokenUsage tokenUsage) {
        if (tokenUsage != null) {
            runContext.metric(Counter.of("input.token.count", "Large Language Model (LLM) input token count", orZero(tokenUsage.getInputTokenCount())));
            runContext.metric(Counter.of("output.token.count", "Large Language Model (LLM) output token count", orZero(tokenUsage.getOutputTokenCount())));
            runContext.metric(Counter.of("total.token.count", "Large Language Model (LLM) total token count", orZero(tokenUsage.getTotalTokenCount())));
        }
    }

    public static Map<String, Object> parseJson(String json) throws JsonProcessingException {
        if (json == null) {
            return null;
        }

        // some LLM may return an empty string so we handle it specifically
        if (json.isBlank()) {
            return Collections.emptyMap();
        }

        return JacksonMapper.toMap(json);
    }

    private static Integer orZero(Integer value) {
        return value != null ? value : 0;
    }
}
