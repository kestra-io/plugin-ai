package io.kestra.plugin.ai.domain;

import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.tool.internal.JsonObjectSchemaTranslator;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.annotation.Nullable;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;

import java.util.Map;

@Getter
@Builder
public class ChatConfiguration {
    @Schema(
        title = "Temperature",
        description = "Controls randomness in generation. Typical range is 0.0–1.0. Lower values (e.g., 0.2) make outputs more focused and deterministic, while higher values (e.g., 0.7–1.0) increase creativity and variability."
    )
    private Property<Double> temperature;

    @Schema(
        title = "Top-K",
        description = "Limits sampling to the top K most likely tokens at each step. Typical values are between 20 and 100. Smaller values reduce randomness; larger values allow more diverse outputs."
    )
    private Property<Integer> topK;

    @Schema(
        title = "Top-P (nucleus sampling)",
        description = "Selects from the smallest set of tokens whose cumulative probability is ≤ topP. Typical values are 0.8–0.95. Lower values make the output more focused, higher values increase diversity."
    )
    private Property<Double> topP;

    @Schema(
        title = "Seed",
        description = "Optional random seed for reproducibility. Provide a positive integer (e.g., 42, 1234). Using the same seed with identical settings produces repeatable outputs."
    )
    private Property<Integer> seed;

    @Schema(
        title = "Log LLM requests",
        description = "If true, prompts and configuration sent to the LLM will be logged at INFO level."
    )
    private Property<Boolean> logRequests;

    @Schema(
        title = "Log LLM responses",
        description = "If true, raw responses from the LLM will be logged at INFO level."
    )
    private Property<Boolean> logResponses;

    @Schema(
        title = "Response format",
        description = """
            Defines the expected output format. Default is plain text.
            Some providers allow requesting JSON or schema-constrained outputs, but support varies and may be incompatible with tool use.
            When using a JSON schema, the output will be returned under the key `jsonOutput`."""
    )
    private ResponseFormat responseFormat;

    @Schema(
        title = "Enable Thinking",
        description = """
            Enables internal reasoning ('thinking') in supported language models, allowing the model to perform intermediate reasoning steps
            before producing a final output; this is useful for complex tasks like multi-step problem solving or decision making, but may
            increase token usage and response time, and is only applicable to compatible models."""
    )
    private Property<Boolean> thinkingEnabled;

    @Schema(
        title = "Thinking Token Budget",
        description = """
            Specifies the maximum number of tokens allocated as a budget for internal reasoning processes, such as generating intermediate
            thoughts or chain-of-thought sequences, allowing the model to perform multi-step reasoning before producing the final output."""
    )
    private Property<Integer> thinkingBudgetTokens;

    @Schema(
        title = "Return Thinking",
        description = """
            Controls whether to return the model's internal reasoning or 'thinking' text, if available. When enabled,
            the reasoning content is extracted from the response and made available in the AiMessage object.
            It Does not trigger the thinking process itself—only affects whether the output is parsed and returned."""
    )
    private Property<Boolean> returnThinking;

    @Schema(
        description = "Maximum number of tokens the model can generate in the completion (response). This limits the length of the output.",
        example = "1024"
    )
    @Nullable
    private Property<Integer> maxToken;

    public dev.langchain4j.model.chat.request.ResponseFormat computeResponseFormat(RunContext runContext) throws IllegalVariableEvaluationException {
        if (responseFormat == null) {
            return dev.langchain4j.model.chat.request.ResponseFormat.TEXT;
        }

        return responseFormat.to(runContext);
    }

    public static ChatConfiguration empty() {
        return ChatConfiguration.builder().build();
    }

    @Getter
    @Builder
    public static class ResponseFormat {
        @Schema(
            title = "Response format type",
            description = """
                Specifies how the LLM should return output.
                Allowed values:
                - TEXT (default): free-form natural language.
                - JSON: structured output validated against a JSON Schema.
                """
        )
        @NotNull
        @Builder.Default
        private Property<ResponseFormatType> type = Property.ofValue(ResponseFormatType.TEXT);

        @Schema(
            title = "JSON Schema (used when type = JSON)",
            description = """
                Provide a JSON Schema describing the expected structure of the response.
                In Kestra flows, define the schema in YAML (it is still a JSON Schema object).
                Example (YAML):
                ```yaml
                  responseFormat:
                    type: JSON
                    jsonSchema:
                      type: object
                      required: ["category", "priority"]
                      properties:
                        category:
                          type: string
                          enum: ["ACCOUNT", "BILLING", "TECHNICAL", "GENERAL"]
                        priority:
                          type: string
                          enum: ["LOW", "MEDIUM", "HIGH"]
                ```
                Note: Provider support for strict schema enforcement varies. If unsupported,
                guide the model about the expected output structure via the prompt and validate downstream.
                """
        )
        private Property<Map<String, Object>> jsonSchema;

        @Schema(
            title = "Schema description (optional)",
            description = """
                Natural-language description of the schema to help the model produce the right fields.
                Example: "Classify a customer ticket into category and priority."
                """
        )
        private Property<String> jsonSchemaDescription;

        dev.langchain4j.model.chat.request.ResponseFormat to(RunContext runContext) throws IllegalVariableEvaluationException {
            var responseFormatType = runContext.render(type).as(ResponseFormatType.class).orElse(ResponseFormatType.TEXT);
            if (responseFormatType == ResponseFormatType.TEXT && jsonSchema != null) {
                throw new IllegalArgumentException("`jsonSchema` property is only allowed when `type` is `JSON`");
            }

            JsonSchema langchain4jJsonSchema = null;
            if (jsonSchema != null) {
                JsonObjectSchema jsonObjectSchema = JsonObjectSchemaTranslator.fromOpenAPISchema(runContext.render(jsonSchema).asMap(String.class, Object.class), runContext.render(jsonSchemaDescription).as(String.class).orElse(null));
                langchain4jJsonSchema = JsonSchema.builder().name("output").rootElement(jsonObjectSchema).build();
            }
            return dev.langchain4j.model.chat.request.ResponseFormat.builder()
                .type(runContext.render(type).as(ResponseFormatType.class).orElse(ResponseFormatType.TEXT))
                .jsonSchema(langchain4jJsonSchema)
                .build();
        }
    }
}
