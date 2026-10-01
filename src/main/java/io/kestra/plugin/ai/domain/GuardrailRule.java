package io.kestra.plugin.ai.domain;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Builder;
import lombok.Getter;
import lombok.extern.jackson.Jacksonized;
import io.kestra.core.models.annotations.PluginProperty;

@Getter
@Builder
@Jacksonized
public class GuardrailRule {

    @Schema(
        title = "Pebble expression",
        description = "Condition that must evaluate to `true` for the guardrail to pass. Input guardrails can read `message` (the user message text); output guardrails can read `response` (the AI response text), `finishReason`, `inputTokenCount` and `outputTokenCount`. No default: this property is required and must not be blank.",
        example = "{{ not (response contains 'CONFIDENTIAL') }}"
    )
    @NotBlank
    @PluginProperty(group = "advanced")
    private String expression;

    @Schema(
        title = "Violation message",
        description = "Text reported in the task output when the expression evaluates to `false`. No default: this property is required and must not be blank.",
        example = "Response leaked confidential content."
    )
    @NotBlank
    @PluginProperty(group = "advanced")
    private String message;
}
