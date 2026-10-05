package io.kestra.plugin.ai.completion;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Metric;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.AIUtils;
import io.kestra.plugin.ai.TokenBudgetChatModel;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.Guardrails;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.ai.domain.TokenUsage;
import io.kestra.plugin.ai.guardrail.GuardrailsEvaluator;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.annotation.Nullable;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.experimental.SuperBuilder;
import org.slf4j.Logger;

import java.time.Duration;
import java.util.List;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Extract JSON fields from text",
    description = """
        Builds a JSON schema from `jsonFields` (all required) and asks the model to return compliant JSON for the given input (`prompt` or `contentBlocks`). Requires a provider that supports JSON response formats; otherwise include schema hints in the prompt. Returns extracted JSON, token usage, and finish reason."""
)
@Plugin(
    examples = {
        @Example(
            title = "Extract person fields (Gemini)",
            full = true,
            code = {
                """
                    id: json_structured_extraction
                    namespace: company.ai

                    tasks:
                      - id: extract_person
                        type: io.kestra.plugin.ai.completion.JSONStructuredExtraction
                        schemaName: Person
                        jsonFields:
                          - name
                          - city
                          - country
                          - email
                        prompt: |
                          From the text below, extract the person's name, city, and email.
                          If a field is missing, leave it blank.

                          Text:
                          "Hi! I'm John Smith from Paris, France. You can reach me at john.smith@example.com."
                        systemMessage: You extract structured data in JSON format.
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                          modelName: gemini-3.5-flash-lite
                    """
            }
        ),
        @Example(
            title = "Extract order details (OpenAI)",
            full = true,
            code = {
                """
                    id: json_structured_extraction_order
                    namespace: company.ai

                    tasks:
                      - id: extract_order
                        type: io.kestra.plugin.ai.completion.JSONStructuredExtraction
                        schemaName: Order
                        jsonFields:
                          - order_id
                          - customer_name
                          - city
                          - total_amount
                        prompt: |
                          Extract the order_id, customer_name, city, and total_amount from the message.
                          For the total amount, keep only the number without the currency symbol.
                          Return only JSON with the requested keys.

                          Message:
                          "Order #A-1043 for Jane Doe, shipped to Berlin. Total: 249.99 EUR."
                        systemMessage: You are a precise JSON data extraction assistant.
                        provider:
                          type: io.kestra.plugin.ai.provider.OpenAI
                          apiKey: "{{ secret('OPENAI_API_KEY') }}"
                          modelName: gpt-5-mini
                        guardrails:
                          input:
                            - expression: "{{ message.length < 10000 }}"
                              message: "Message too long"
                          output:
                            - expression: "{{ not (response contains 'CONFIDENTIAL') }}"
                              message: "Response contains confidential information"

                    """
            }
        )
    },
    metrics = {
        @Metric(
            name = "input.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) input token count"
        ),
        @Metric(
            name = "output.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) output token count"
        ),
        @Metric(
            name = "total.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) total token count"
        ),
        @Metric(
            name = "ai.provider.calls",
            type = Counter.TYPE,
            unit = "calls",
            description = "Number of times a chat model is obtained from a provider, tagged by provider class name"
        )
    },
    aliases = { "io.kestra.plugin.langchain4j.JSONStructuredExtraction", "io.kestra.plugin.langchain4j.completion.JSONStructuredExtraction" }
)
public class JSONStructuredExtraction extends Task implements RunnableTask<JSONStructuredExtraction.Output> {

    @Schema(
        title = "Text prompt",
        description = "Text to extract structured data from. Set either this property or `contentBlocks`, not both.",
        example = "{{ inputs.invoice_text }}"
    )
    @Nullable
    @PluginProperty(group = "main")
    private Property<String> prompt;

    @Schema(
        title = "Content blocks",
        description = "Multimodal input to extract from, as a list of `TEXT`, `IMAGE` or `PDF` blocks. Set either this property or `prompt`, not both. For `IMAGE` and `PDF` blocks, the `uri` supports the `kestra://`, `file://` and `nsfile://` schemes.",
        example = "[{type: \"PDF\", uri: \"{{ inputs.invoice }}\"}]"
    )
    @Nullable
    private Property<List<ChatMessage.ContentBlock>> contentBlocks;

    @Schema(
        title = "System message",
        description = "Instruction steering how the model extracts the fields. Defaults to `You are a structured JSON extraction assistant. Always respond with valid JSON.`.",
        example = "You are a structured JSON extraction assistant. Always respond with valid JSON."
    )
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<String> systemMessage = Property.ofValue(
        "You are a structured JSON extraction assistant. Always respond with valid JSON."
    );

    @Schema(
        title = "Schema name",
        description = "Name given to the JSON schema the model fills in, which helps the model understand what it is extracting. No default: this property is required.",
        example = "invoice"
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<String> schemaName;

    @Schema(
        title = "JSON fields",
        description = "Field names to extract from the input, which become the keys of the returned JSON object. No default: this property is required.",
        example = "[\"invoice_number\", \"total_amount\", \"due_date\"]"
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<List<String>> jsonFields;

    @Schema(
        title = "Language model provider",
        description = "Model provider that performs the extraction. No default: this property is required.",
        example = "{type: \"io.kestra.plugin.ai.provider.GoogleGemini\", apiKey: \"{{ secret('GEMINI_API_KEY') }}\", modelName: \"gemini-3.5-flash-lite\"}"
    )
    @NotNull
    @PluginProperty(group = "main")
    private ModelProvider provider;

    @Schema(
        title = "Chat configuration",
        description = "Chat model settings (temperature, token limits, and so on). Defaults to an empty configuration, so the provider's own defaults apply; a low temperature gives the most consistent extractions.",
        example = "{temperature: 0.1}"
    )
    @NotNull
    @PluginProperty(group = "advanced")
    @Builder.Default
    private ChatConfiguration configuration = ChatConfiguration.empty();

    @Schema(
        title = "Guardrails",
        description = "Rules validating the call: input guardrails run against the prompt before the LLM is called, output guardrails against the extracted JSON before it is returned. The first failing rule stops execution and sets `guardrailViolated` to `true` in the output. Not set by default.",
        example = "{input: [{type: \"io.kestra.plugin.ai.guardrail.ExpressionInputGuardrail\", expression: \"{{ prompt | length < 5000 }}\"}]}"
    )
    @PluginProperty(group = "advanced")
    @Nullable
    private Guardrails guardrails;

    @Override
    public JSONStructuredExtraction.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        String rPrompt = prompt == null ? null : runContext.render(prompt).as(String.class).orElse(null);
        List<ChatMessage.ContentBlock> rContentBlocks = contentBlocks == null ? null : runContext.render(contentBlocks).asList(ChatMessage.ContentBlock.class);
        CompletionInputContentUtils.validatePromptInput("JSONStructuredExtraction", rPrompt, rContentBlocks);

        String rSchemaName = runContext.render(schemaName).as(String.class).orElseThrow();
        List<String> rJsonFields = Property.asList(jsonFields, runContext, String.class);

        String rSystemMessage = runContext.render(systemMessage).as(String.class).orElseThrow();

        // Input guardrail check
        String inputViolation = GuardrailsEvaluator.checkInput(guardrails, rPrompt, runContext);
        if (inputViolation != null) {
            return buildGuardrailViolationOutput(logger, "Input guardrail violated: {}", inputViolation, "Input guardrail: ");
        }

        ResponseFormat responseFormat = ResponseFormat.builder()
            .type(ResponseFormatType.JSON)
            .jsonSchema(
                JsonSchema.builder()
                    .name(rSchemaName)
                    .rootElement(buildDynamicSchema(rJsonFields))
                    .build()
            )
            .build();

        ChatRequest chatRequest = ChatRequest.builder()
            .parameters(ChatRequestParameters.builder()
                .responseFormat(responseFormat)
                .build())
            .messages(List.of(
                SystemMessage.systemMessage(rSystemMessage),
                CompletionInputContentUtils.toUserMessage(runContext, rPrompt, rContentBlocks)
            ))
            .build();

        Duration taskTimeout = runContext.render(this.getTimeout()).as(Duration.class).orElse(Duration.ofSeconds(120));
        ChatModel model = TokenBudgetChatModel.wrap(
            this.provider.chatModel(runContext, configuration, taskTimeout),
            runContext,
            configuration
        );
        runContext.metric(Counter.of("ai.provider.calls", 1, "provider", this.provider.getClass().getName()));

        ChatResponse answer = model.chat(chatRequest);
        logger.debug("Generated Structured Extraction: {}", answer.aiMessage().text());

        // Output guardrail check
        String outputViolation = GuardrailsEvaluator.checkOutput(guardrails, answer, runContext);
        if (outputViolation != null) {
            return buildGuardrailViolationOutput(logger, "Output guardrail violated: {}", outputViolation, "Output guardrail: ");
        }

        TokenUsage tokenUsage = TokenUsage.from(answer.tokenUsage());
        AIUtils.sendMetrics(runContext, tokenUsage);

        return Output.builder()
            .schemaName(rSchemaName)
            .extractedJson(answer.aiMessage().text())
            .tokenUsage(tokenUsage)
            .finishReason(answer.finishReason())
            .build();
    }

    private static Output buildGuardrailViolationOutput(Logger logger, String s, String outputViolation, String x) {
        logger.warn(s, outputViolation);
        return Output.builder()
            .guardrailViolated(true)
            .guardrailViolationMessage(x + outputViolation)
            .build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(
            title = "Schema name",
            description = "Name of the JSON schema that was used for the extraction, echoed back from the input property.",
            example = "invoice"
        )
        private String schemaName;

        @Schema(
            title = "Extracted JSON",
            description = "Structured JSON object the model produced, with one key per entry of `jsonFields`.",
            example = "{\"invoice_number\": \"INV-2026-014\", \"total_amount\": \"1240.00\", \"due_date\": \"2026-02-15\"}"
        )
        private String extractedJson;

        @Schema(
            title = "Token usage",
            description = "Input, output and total tokens billed for the call, when the provider reports them.",
            example = "{inputTokenCount: 320, outputTokenCount: 48, totalTokenCount: 368}"
        )
        private TokenUsage tokenUsage;

        @Schema(
            title = "Finish reason",
            description = "Why the model stopped generating, such as `STOP`, `LENGTH` or `CONTENT_FILTER`, when the provider reports it.",
            example = "STOP"
        )
        private FinishReason finishReason;

        @Schema(
            title = "Guardrail violated",
            description = "Whether a guardrail rule rejected the input or the output. `false` when every rule passed.",
            example = "false"
        )
        @Builder.Default
        private boolean guardrailViolated = false;

        @Schema(
            title = "Guardrail violation message",
            description = "Message from the first guardrail rule that failed. Empty when no rule was violated.",
            example = "Prompt exceeds the allowed length."
        )
        private String guardrailViolationMessage;
    }

    public static JsonObjectSchema buildDynamicSchema(List<String> fields) {
        JsonObjectSchema.Builder schemaBuilder = JsonObjectSchema.builder();
        fields.forEach(schemaBuilder::addStringProperty);
        schemaBuilder.required(fields);
        return schemaBuilder.build();
    }

}
