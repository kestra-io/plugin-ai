package io.kestra.plugin.ai.guardrail;

import org.slf4j.Logger;

import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.ListUtils;
import io.kestra.plugin.ai.domain.Guardrails;

import dev.langchain4j.guardrail.InputGuardrailException;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.service.AiServices;
import jakarta.annotation.Nullable;

public final class GuardrailsEvaluator {

    private GuardrailsEvaluator() {
    }

    public static String checkInput(@Nullable Guardrails guardrails, String prompt, RunContext ctx) {
        if (guardrails == null || ListUtils.emptyOnNull(guardrails.getInput()).isEmpty())
            return null;
        return new ExpressionInputGuardrail(guardrails.getInput(), ctx).checkOrNull(prompt);
    }

    public static String checkOutput(@Nullable Guardrails guardrails, ChatResponse response, RunContext ctx) {
        if (guardrails == null || ListUtils.emptyOnNull(guardrails.getOutput()).isEmpty())
            return null;
        return new ExpressionOutputGuardrail(guardrails.getOutput(), ctx).checkOrNull(response);
    }

    /**
     * Registers input/output guardrails on an AiServices builder when rules are configured.
     */
    public static void applyGuardrails(@Nullable Guardrails guardrails, AiServices<?> builder, RunContext ctx) {
        if (guardrails == null)
            return;
        if (!ListUtils.emptyOnNull(guardrails.getInput()).isEmpty()) {
            builder.inputGuardrails(new ExpressionInputGuardrail(guardrails.getInput(), ctx));
        }
        if (!ListUtils.emptyOnNull(guardrails.getOutput()).isEmpty()) {
            builder.outputGuardrails(new ExpressionOutputGuardrail(guardrails.getOutput(), ctx));
        }
    }

    /**
     * Logs the guardrail violation and returns the formatted violation message.
     * Shared by the catch blocks in AiServices-based tasks.
     */
    public static String logAndFormatViolation(Exception e, Logger logger) {
        String guardrailType = e instanceof InputGuardrailException ? "Input" : "Output";
        logger.warn("{} guardrail violated: {}", guardrailType, e.getMessage());
        return guardrailType + " guardrail: " + e.getMessage();
    }
}
