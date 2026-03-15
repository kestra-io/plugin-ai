package io.kestra.plugin.ai.guardrail;

import java.util.List;
import java.util.Map;

import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.GuardrailRule;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.guardrail.InputGuardrail;
import dev.langchain4j.guardrail.InputGuardrailResult;

/**
 * LangChain4j {@link InputGuardrail} that evaluates a list of Kestra Pebble expressions
 * against the user message before it is sent to the LLM.
 *
 * <p>
 * The expression context exposes a single variable:
 * <ul>
 * <li>{@code message} – the text of the user message</li>
 * </ul>
 *
 * <p>
 * Expressions must evaluate to {@code true} for the guardrail to pass.
 * The first failing expression triggers a {@code fatal} result, which causes
 * LangChain4j to throw an {@link dev.langchain4j.guardrail.InputGuardrailException}
 * that the task catches and converts into a {@code guardrailViolated} output.
 */
public class ExpressionInputGuardrail implements InputGuardrail {

    private final List<GuardrailRule> rules;
    private final RunContext runContext;

    public ExpressionInputGuardrail(List<GuardrailRule> rules, RunContext runContext) {
        this.rules = rules;
        this.runContext = runContext;
    }

    @Override
    public InputGuardrailResult validate(UserMessage userMessage) {
        String messageText = userMessage.singleText();
        Map<String, Object> context = Map.of("message", messageText);

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
