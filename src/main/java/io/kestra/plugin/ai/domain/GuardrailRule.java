package io.kestra.plugin.ai.domain;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Builder;
import lombok.Getter;
import lombok.extern.jackson.Jacksonized;

@Getter
@Builder
@Jacksonized
public class GuardrailRule {

    @Schema(
        title = "Pebble expression",
        description = """
            A Pebble expression that must evaluate to `true` for the guardrail to pass.
            For input guardrails, the variable `message` contains the user message text.
            For output guardrails, the variable `response` contains the AI response text,
            `finishReason` contains the finish reason, `inputTokenCount` and `outputTokenCount`
            contain the respective token counts.
            Example: `{{ message.length < 10000 }}`
            Example: `{{ not (response contains 'CONFIDENTIAL') }}`
            """
    )
    @NotBlank
    private String expression;

    @Schema(
        title = "Violation message",
        description = "The message returned when the expression evaluates to `false`."
    )
    @NotBlank
    private String message;
}
