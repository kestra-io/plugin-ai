package io.kestra.plugin.ai.completion;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
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
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.ai.domain.TokenUsage;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;
import org.slf4j.Logger;

import java.util.List;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Extract JSON fields from text",
    description = """
        Builds a JSON schema from `jsonFields` (all required) and asks the model to return compliant JSON for the given prompt. Requires a provider that supports JSON response formats; otherwise include schema hints in the prompt. Returns extracted JSON, token usage, and finish reason."""
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
                          modelName: gemini-2.5-flash
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
        )
    },
    aliases = {"io.kestra.plugin.langchain4j.JSONStructuredExtraction", "io.kestra.plugin.langchain4j.completion.JSONStructuredExtraction"}
)
public class JSONStructuredExtraction extends Task implements RunnableTask<JSONStructuredExtraction.Output> {

    @Schema(title = "Text prompt", description = "The input text for structured JSON extraction.")
    private Property<String> prompt;

    @Schema(title = "System message", description = "Optional system instruction for the model.")
    @Builder.Default
    private Property<String> systemMessage = Property.ofValue(
        "You are a structured JSON extraction assistant. Always respond with valid JSON."
    );

    @Schema(title = "Schema Name", description = "The name of the JSON schema for structured extraction")
    @NotNull
    private Property<String> schemaName;

    @Schema(title = "JSON Fields", description = "List of fields to extract from the text")
    @NotNull
    private Property<List<String>> jsonFields;

    @Schema(title = "Language Model Provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @Schema(title = "Chat configuration")
    @NotNull
    @PluginProperty
    @Builder.Default
    private ChatConfiguration configuration = ChatConfiguration.empty();

    @Override
    public JSONStructuredExtraction.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        String rPrompt = runContext.render(prompt).as(String.class).orElseThrow(() ->
            new IllegalArgumentException("Prompt must be provided for structured extraction"));

        String rSchemaName = runContext.render(schemaName).as(String.class).orElseThrow();
        List<String> rJsonFields = Property.asList(jsonFields, runContext, String.class);

        String rSystemMessage = runContext.render(systemMessage).as(String.class).orElseThrow();

        ResponseFormat responseFormat = ResponseFormat.builder()
            .type(ResponseFormatType.JSON)
            .jsonSchema(JsonSchema.builder()
                .name(rSchemaName)
                .rootElement(buildDynamicSchema(rJsonFields))
                .build())
            .build();

        ChatRequest chatRequest = ChatRequest.builder()
            .parameters(ChatRequestParameters.builder()
                .responseFormat(responseFormat)
                .build())
            .messages(List.of(
                SystemMessage.systemMessage(rSystemMessage),
                UserMessage.userMessage(rPrompt)
            ))
            .build();

        ChatModel model = this.provider.chatModel(runContext, configuration);

        ChatResponse answer = model.chat(chatRequest);
        logger.debug("Generated Structured Extraction: {}", answer.aiMessage().text());

        TokenUsage tokenUsage = TokenUsage.from(answer.tokenUsage());
        AIUtils.sendMetrics(runContext, tokenUsage);

        return Output.builder()
            .schemaName(rSchemaName)
            .extractedJson(answer.aiMessage().text())
            .tokenUsage(tokenUsage)
            .finishReason(answer.finishReason())
            .build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Schema Name", description = "The schema name used for the structured JSON extraction")
        private String schemaName;

        @Schema(title = "Extracted JSON", description = "The structured JSON output")
        private String extractedJson;

        @Schema(title = "Token usage")
        private TokenUsage tokenUsage;

        @Schema(title = "Finish reason")
        private FinishReason finishReason;
    }

    public static JsonObjectSchema buildDynamicSchema(List<String> fields) {
        JsonObjectSchema.Builder schemaBuilder = JsonObjectSchema.builder();
        fields.forEach(schemaBuilder::addStringProperty);
        schemaBuilder.required(fields);
        return schemaBuilder.build();
    }
}
