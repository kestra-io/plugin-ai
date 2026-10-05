package io.kestra.plugin.ai.domain;

import java.util.List;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import lombok.extern.jackson.Jacksonized;
import io.kestra.core.models.annotations.PluginProperty;

@Getter
@Builder
@Jacksonized
public class Guardrails {

    @Schema(
        title = "Input guardrails",
        description = "Rules evaluated against the user message before it reaches the LLM. Each rule's Pebble expression can read the `message` variable holding the user message text. The first failing rule halts execution and reports a guardrail violation in the task output. Not set by default.",
        example = "[{expression: \"{{ message.length < 10000 }}\", message: \"Prompt is too long.\"}]"
    )
    @PluginProperty(group = "advanced")
    private List<GuardrailRule> input;

    @Schema(
        title = "Output guardrails",
        description = "Rules evaluated against the AI response before it is returned. Each rule's Pebble expression can read `response` (the response text), `finishReason`, `inputTokenCount` and `outputTokenCount`. The first failing rule halts execution and reports a guardrail violation in the task output. Not set by default.",
        example = "[{expression: \"{{ not (response contains 'CONFIDENTIAL') }}\", message: \"Response leaked confidential content.\"}]"
    )
    @PluginProperty(group = "advanced")
    private List<GuardrailRule> output;
}
