package io.kestra.plugin.ai.completion;

import dev.langchain4j.data.message.AiMessage;
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
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.AIUtils;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
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
    title = "Extract structured JSON with LLMs",
    description = "Convert unstructured text into a JSON object with predefined fields. Provide a schema name and the list of fields to extract. Compatible with OpenAI, Gemini, and Ollama."
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
                    messages:
                      - type: SYSTEM
                        content: You extract structured data in JSON format.
                      - type: USER
                        content: |
                          From the text below, extract the person's name, city, and email.
                          If a field is missing, leave it blank.

                          Text:
                          "Hi! I'm John Smith from Paris, France. You can reach me at john.smith@example.com."
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                      modelName: gemini-2.0-flash
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
                    messages:
                      - type: SYSTEM
                        content: You are a precise JSON data extraction assistant.
                      - type: USER
                        content: |
                          Extract the order_id, customer_name, city, and total_amount from the message.
                          For the total amount, keep only the number without the currency symbol.
                          Return only JSON with the requested keys.

                          Message:
                          "Order #A-1043 for Jane Doe, shipped to Berlin. Total: 249.99 EUR."
                    provider:
                      type: io.kestra.plugin.ai.provider.OpenAI
                      apiKey: "{{ kv('OPENAI_API_KEY') }}"
                      modelName: gpt-4o-mini
                """
            }
        )
    },
    aliases = {"io.kestra.plugin.langchain4j.JSONStructuredExtraction", "io.kestra.plugin.langchain4j.completion.JSONStructuredExtraction"}
)
public class JSONStructuredExtraction extends Task implements RunnableTask<JSONStructuredExtraction.Output> {

    @Schema(
        title = "Chat Messages",
        description = "The list of chat messages for the extraction context. There can be only one system message, and the last message must be a user message."
    )
    @NotNull
    private Property<List<ChatMessage>> messages;

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

        // Render inputs
        List<ChatMessage> rMessages = runContext.render(messages).asList(ChatMessage.class);
        String rSchemaName = runContext.render(schemaName).as(String.class).orElseThrow();
        List<String> rJsonFields = Property.asList(jsonFields, runContext, String.class);

        if (rMessages.isEmpty()) {
            throw new IllegalArgumentException("At least one user message must be provided for structured extraction");
        }

        // Convert messages to LangChain4j messages
        List<dev.langchain4j.data.message.ChatMessage> chatMessages = rMessages.stream()
            .map(msg -> switch (msg.type()) {
                case SYSTEM -> SystemMessage.systemMessage(msg.content());
                case USER -> UserMessage.userMessage(msg.content());
                case AI -> AiMessage.aiMessage(msg.content());
            })
            .toList();

        // Build JSON schema
        ResponseFormat responseFormat = ResponseFormat.builder()
            .type(ResponseFormatType.JSON)
            .jsonSchema(JsonSchema.builder()
                .name(rSchemaName)
                .rootElement(buildDynamicSchema(rJsonFields))
                .build())
            .build();

        // Build chat request
        ChatRequest chatRequest = ChatRequest.builder()
            .parameters(ChatRequestParameters.builder()
                .responseFormat(responseFormat)
                .build())
            .messages(chatMessages)
            .build();

        // Get model
        ChatModel model = this.provider.chatModel(runContext, configuration);

        // Execute
        ChatResponse answer = model.chat(chatRequest);
        logger.debug("Generated Structured Extraction: {}", answer.aiMessage().text());

        // Send metrics for token usage
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
