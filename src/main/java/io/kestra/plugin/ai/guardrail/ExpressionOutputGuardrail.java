package io.kestra.plugin.ai.guardrail;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.GuardrailRule;

import dev.langchain4j.guardrail.OutputGuardrail;
import dev.langchain4j.guardrail.OutputGuardrailRequest;
import dev.langchain4j.guardrail.OutputGuardrailResult;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;

/**
 * LangChain4j {@link OutputGuardrail} that evaluates a list of Kestra Pebble expressions
 * against the AI response before it is returned to the caller.
 *
 * <p>
 * The expression context exposes:
 * <ul>
 * <li>{@code response} – the AI response text</li>
 * <li>{@code finishReason} – e.g. {@code "STOP"}, {@code "LENGTH"}</li>
 * <li>{@code inputTokenCount} – number of input tokens</li>
 * <li>{@code outputTokenCount} – number of output tokens</li>
 * </ul>
 *
 * <p>
 * Expressions must evaluate to {@code true} for the guardrail to pass.
 * The first failing expression triggers a {@code fatal} result.
 */
public class ExpressionOutputGuardrail implements OutputGuardrail {

    private final List<GuardrailRule> rules;
    private final RunContext runContext;

    public ExpressionOutputGuardrail(List<GuardrailRule> rules, RunContext runContext) {
        this.rules = rules;
        this.runContext = runContext;
    }

    @Override
    public OutputGuardrailResult validate(OutputGuardrailRequest request) {
        ChatResponse chatResponse = request.responseFromLLM();
        String responseText = chatResponse.aiMessage().text();
        FinishReason finishReason = chatResponse.finishReason();
        TokenUsage tokenUsage = chatResponse.tokenUsage();

        Map<String, Object> context = new HashMap<>();
        context.put("response", responseText != null ? responseText : "");
        context.put("finishReason", finishReason != null ? finishReason.name() : "");
        context.put("inputTokenCount", tokenUsage != null && tokenUsage.inputTokenCount() != null ? tokenUsage.inputTokenCount() : 0);
        context.put("outputTokenCount", tokenUsage != null && tokenUsage.outputTokenCount() != null ? tokenUsage.outputTokenCount() : 0);

        for (GuardrailRule rule : rules) {
            try {
                String result = runContext.render(Property.<String> ofExpression(rule.getExpression()))
                    .as(String.class, context)
                    .orElse("false");
                if (!Boolean.parseBoolean(result.trim())) {
                    return fatal(rule.getMessage());
                }
            } catch (Exception e) {
                return fatal("Guardrail expression evaluation failed: " + e.getMessage());
            }
        }

        return success();
    }
}
