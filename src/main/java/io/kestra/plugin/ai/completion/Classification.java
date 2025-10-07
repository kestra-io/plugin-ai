package io.kestra.plugin.ai.completion;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
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

import java.util.ArrayList;
import java.util.List;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Text categorization with LLMs",
    description = "Categorize text into one of the provided classes using an LLM."
)
@Plugin(
    examples = {
        @Example(
            title = "Perform sentiment analysis of product reviews",
            full = true,
            code = {
                """
                    id: text_categorization
                    namespace: company.ai

                    tasks:
                      - id: categorize
                        type: io.kestra.plugin.ai.completion.Classification
                        messages:
                          - type: SYSTEM
                            content: You are a sentiment analysis assistant. Classify text as positive, negative or neutral.
                          - type: USER
                            content: "I absolutely love this product, it's fantastic!"
                        classes:
                          - positive
                          - negative
                          - neutral
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          apiKey: "{{ kv('GEMINI_API_KEY') }}"
                          modelName: gemini-2.0-flash
                    """
            }
        )
    },
    aliases = {"io.kestra.plugin.langchain4j.Classification", "io.kestra.plugin.langchain4j.completion.Classification"}
)
public class Classification extends Task implements RunnableTask<Classification.Output> {

    @Schema(
        title = "Chat Messages",
        description = "The list of chat messages for the current conversation. There can be only one system message, and the last message must be a user message."
    )
    @NotNull
    private Property<List<ChatMessage>> messages;

    @Schema(title = "Classification Options", description = "The list of possible classification categories")
    @NotNull
    private Property<List<String>> classes;

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
    public Classification.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        // Render messages and classes
        List<io.kestra.plugin.ai.domain.ChatMessage> rMessages =
            runContext.render(messages).asList(io.kestra.plugin.ai.domain.ChatMessage.class);
        List<String> rClasses = runContext.render(classes).asList(String.class);

        if (rMessages.isEmpty()) {
            throw new IllegalArgumentException("At least one user message must be provided");
        }

        // Convert provided messages to LangChain4j format
        List<dev.langchain4j.data.message.ChatMessage> chatMessages = new ArrayList<>();

        // Add system message for classification instruction only if classes are provided
        if (rClasses != null && !rClasses.isEmpty()) {
            chatMessages.add(SystemMessage.systemMessage(
                "Respond by only one of the following classes by typing just the exact class name: " + rClasses
            ));
        }

        // Add all provided messages
        chatMessages.addAll(
            rMessages.stream()
                .map(dto -> switch (dto.type()) {
                    case SYSTEM -> SystemMessage.systemMessage(dto.content());
                    case AI -> AiMessage.aiMessage(dto.content());
                    case USER -> UserMessage.userMessage(dto.content());
                })
                .toList()
        );

        // Validate structure
        long nbSystem = chatMessages.stream()
            .filter(msg -> msg.type() == dev.langchain4j.data.message.ChatMessageType.SYSTEM)
            .count();
        if (nbSystem > 2) { // counting the one for classes
            throw new IllegalArgumentException("Only one system message is allowed");
        }
        if (chatMessages.getLast().type() != dev.langchain4j.data.message.ChatMessageType.USER) {
            throw new IllegalArgumentException("The last message must be a user message");
        }

        // Get model
        ChatModel model = this.provider.chatModel(runContext, configuration);

        // Perform classification
        ChatResponse response = model.chat(chatMessages);
        logger.debug("Generated Classification: {}", response.aiMessage().text());

        // Send token usage metrics
        TokenUsage tokenUsage = TokenUsage.from(response.tokenUsage());
        AIUtils.sendMetrics(runContext, tokenUsage);

        return Output.builder()
            .classification(response.aiMessage().text())
            .tokenUsage(tokenUsage)
            .finishReason(response.finishReason())
            .build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Classification Result", description = "The classified category of the input text")
        private final String classification;

        @Schema(title = "Token usage")
        private TokenUsage tokenUsage;

        @Schema(title = "Finish reason")
        private FinishReason finishReason;
    }
}
