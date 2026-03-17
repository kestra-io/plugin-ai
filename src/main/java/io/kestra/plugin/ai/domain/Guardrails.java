package io.kestra.plugin.ai.domain;

import java.util.List;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import lombok.extern.jackson.Jacksonized;

@Getter
@Builder
@Jacksonized
public class Guardrails {

    @Schema(
        title = "Input guardrails",
        description = """
            Guardrails evaluated against the user message before it is sent to the LLM.
            Each rule's Pebble expression has access to the `message` variable (the user message text).
            The first failing rule halts execution and returns a guardrail violation in the task output."""
    )
    private List<GuardrailRule> input;

    @Schema(
        title = "Output guardrails",
        description = """
            Guardrails evaluated against the AI response before it is returned.
            Each rule's Pebble expression has access to `response` (the AI response text),
            `finishReason`, `inputTokenCount`, and `outputTokenCount`.
            The first failing rule halts and returns a guardrail violation in the task output."""
    )
    private List<GuardrailRule> output;
}
