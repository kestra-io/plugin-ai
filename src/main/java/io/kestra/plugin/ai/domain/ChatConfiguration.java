package io.kestra.plugin.ai.domain;

import java.util.Map;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.tool.internal.JsonObjectSchemaTranslator;

import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.annotation.Nullable;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;
import io.kestra.core.models.annotations.PluginProperty;

@Getter
@Builder
public class ChatConfiguration {
    @Schema(
        title = "Temperature",
        description = "Randomness of the generation, typically between 0.0 and 1.0. Lower values such as 0.2 make outputs focused and repeatable; higher values such as 0.7-1.0 make them more creative and varied. Not set by default, in which case the provider's own default applies.",
        example = "0.7"
    )
    @PluginProperty(group = "advanced")
    private Property<Double> temperature;

    @Schema(
        title = "Top-K",
        description = "Restricts sampling to the K most likely tokens at each step, typically between 20 and 100. Smaller values reduce randomness, larger values allow more diversity. Not set by default, in which case the provider's own default applies.",
        example = "40"
    )
    @PluginProperty(group = "advanced")
    private Property<Integer> topK;

    @Schema(
        title = "Top-P (nucleus sampling)",
        description = "Restricts sampling to the smallest set of tokens whose cumulative probability is at most this value, typically 0.8-0.95. Lower values focus the output, higher values diversify it. Not set by default, in which case the provider's own default applies.",
        example = "0.9"
    )
    @PluginProperty(group = "advanced")
    private Property<Double> topP;

    @Schema(
        title = "Seed",
        description = "Positive integer seeding the sampler, so that the same seed with identical settings reproduces the same output. Not set by default (non-deterministic generation).",
        example = "42"
    )
    @PluginProperty(group = "advanced")
    private Property<Integer> seed;

    @Schema(
        title = "Log LLM requests",
        description = "If `true`, the prompts and configuration sent to the LLM are logged at INFO level. Defaults to `false`.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> logRequests;

    @Schema(
        title = "Log LLM responses",
        description = "If `true`, the raw responses returned by the LLM are logged at INFO level. Defaults to `false`.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> logResponses;

    @Schema(
        title = "Response format",
        description = "Shape of the model's output: free-form text, or JSON constrained by a schema. Defaults to plain text. Provider support for schema-constrained output varies and may be incompatible with tool use; when a JSON schema is used, the result is returned under the `jsonOutput` key.",
        example = "{type: \"JSON\", jsonSchema: {type: \"object\", properties: {category: {type: \"string\"}}}}"
    )
    @PluginProperty(group = "processing")
    private ResponseFormat responseFormat;

    @Schema(
        title = "Enable Thinking",
        description = "If `true`, supported models perform internal reasoning steps before answering, which helps on multi-step problems at the cost of extra tokens and latency. Defaults to `false`. For Google Gemini, when neither this property nor `thinkingBudgetTokens` is set, Gemini 2.x models get an explicit `thinkingBudget` of `0` to keep token usage down, while Gemini 3 and later receive no thinking configuration at all, since they reject a zero budget and always think.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> thinkingEnabled;

    @Schema(
        title = "Thinking Token Budget",
        description = "Maximum number of tokens the model may spend on internal reasoning before producing its final answer. Not set by default. For Google Gemini, when neither this property nor `thinkingEnabled` is set, Gemini 2.x models get a budget of `0` (thinking disabled), while Gemini 3 and later are sent no budget and apply their own; set this property to cap it on those models.",
        example = "1024"
    )
    @PluginProperty(group = "advanced")
    private Property<Integer> thinkingBudgetTokens;

    @Schema(
        title = "Return thinking",
        description = "If `true`, the model's reasoning text is parsed out of the response and exposed in the `thinking` output. It does not trigger thinking by itself. Not set by default, except for Google Gemini, where it defaults to `true` so that `thought_signature` values on function-call parts are captured and re-sent on later requests, preventing tool-call failures on native thinking models.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> returnThinking;

    @Schema(
        title = "Maximum output tokens",
        description = "Upper bound on the number of tokens the model may generate in one response, which caps the output length. Not set by default, in which case the provider's own default applies.",
        example = "1024"
    )
    @Nullable
    @PluginProperty(group = "connection")
    private Property<Integer> maxToken;

    @Schema(
        title = "Maximum cumulative tokens",
        description = "Budget for the total input and output tokens this task's model may consume across all its calls in one task run, including every iteration of the tool loop. Must be at least `1`. Not set by default (no limit). The task fails as soon as a response pushes usage over the budget, so that last response is still billed. Nested sub-agents (`io.kestra.plugin.ai.tool.AIAgent`) and SQL retrievers track their own `configuration.maxCumulativeTokens`, and the configured model must report token usage.",
        example = "100000"
    )
    @Nullable
    @PluginProperty(group = "reliability")
    private Property<@Min(1) Integer> maxCumulativeTokens;

    @Schema(
        title = "Enable prompt caching",
        description = "If `true`, ask the provider to cache system messages and tool definitions across requests, which can markedly cut latency and cost when the same system prompt or tool set is reused. Not set by default. Currently honored by Anthropic only; other providers ignore it silently.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> promptCaching;

    public dev.langchain4j.model.chat.request.ResponseFormat computeResponseFormat(RunContext runContext) throws IllegalVariableEvaluationException {
        if (responseFormat == null) {
            return dev.langchain4j.model.chat.request.ResponseFormat.TEXT;
        }

        return responseFormat.to(runContext);
    }

    public boolean computeStrictJsonMode(RunContext runContext) throws IllegalVariableEvaluationException {
        if (responseFormat == null) {
            return false;
        }

        return responseFormat.strictJsonMode(runContext);
    }

    public static ChatConfiguration empty() {
        return ChatConfiguration.builder().build();
    }

    @Getter
    @Builder
    public static class ResponseFormat {
        @Schema(
            title = "Response format type",
            description = "How the model returns its output: `TEXT` for free-form natural language, or `JSON` for output validated against a JSON schema. Defaults to `TEXT`.",
            example = "JSON"
        )
        @NotNull
        @Builder.Default
        @PluginProperty(group = "main")
        private Property<ResponseFormatType> type = Property.ofValue(ResponseFormatType.TEXT);

        @Schema(
            title = "JSON schema",
            description = "JSON Schema object describing the expected response structure, written as YAML in a flow. Only allowed when `type` is `JSON`. Provider support for strict schema enforcement varies; where it is unsupported, describe the expected shape in the prompt and validate downstream. Not set by default.",
            example = "{type: \"object\", required: [\"category\"], properties: {category: {type: \"string\", enum: [\"ACCOUNT\", \"BILLING\"]}}}"
        )
        @PluginProperty(group = "connection")
        private Property<Map<String, Object>> jsonSchema;

        @Schema(
            title = "Schema description",
            description = "Natural-language explanation of the schema, which helps the model produce the right fields. Not set by default.",
            example = "Classify a customer ticket into category and priority."
        )
        @PluginProperty(group = "advanced")
        private Property<String> jsonSchemaDescription;

        @Schema(
            title = "Enable strict JSON schema mode",
            description = "If `true`, providers that support it enforce the JSON schema strictly instead of treating it as a hint. Only allowed when `type` is `JSON`. Defaults to `false`.",
            example = "true"
        )
        @Builder.Default
        @PluginProperty(group = "advanced")
        private Property<Boolean> strictJson = Property.ofValue(false);

        dev.langchain4j.model.chat.request.ResponseFormat to(RunContext runContext) throws IllegalVariableEvaluationException {
            var responseFormatType = runContext.render(type).as(ResponseFormatType.class).orElse(ResponseFormatType.TEXT);
            if (responseFormatType == ResponseFormatType.TEXT && jsonSchema != null) {
                throw new IllegalArgumentException("`jsonSchema` property is only allowed when `type` is `JSON`");
            }
            if (responseFormatType == ResponseFormatType.TEXT && runContext.render(strictJson).as(Boolean.class).orElse(false)) {
                throw new IllegalArgumentException("`strictJson` property is only allowed when `type` is `JSON`");
            }

            JsonSchema langchain4jJsonSchema = null;
            if (jsonSchema != null) {
                JsonObjectSchema jsonObjectSchema = JsonObjectSchemaTranslator
                    .fromOpenAPISchema(runContext.render(jsonSchema).asMap(String.class, Object.class), runContext.render(jsonSchemaDescription).as(String.class).orElse(null));
                langchain4jJsonSchema = JsonSchema.builder().name("output").rootElement(jsonObjectSchema).build();
            }
            return dev.langchain4j.model.chat.request.ResponseFormat.builder()
                .type(runContext.render(type).as(ResponseFormatType.class).orElse(ResponseFormatType.TEXT))
                .jsonSchema(langchain4jJsonSchema)
                .build();
        }

        boolean strictJsonMode(RunContext runContext) throws IllegalVariableEvaluationException {
            return runContext.render(strictJson).as(Boolean.class).orElse(false);
        }
    }
}
