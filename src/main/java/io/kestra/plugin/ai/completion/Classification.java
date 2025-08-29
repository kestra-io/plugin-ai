package io.kestra.plugin.ai.completion;

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
                    prompt: "Categorize the sentiment of: I love this product!"
                    classes:
                      - positive
                      - negative
                      - neutral
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash
                """
            }
        )
    },
    aliases = {"io.kestra.plugin.langchain4j.Classification", "io.kestra.plugin.langchain4j.completion.Classification"}
)
public class Classification extends Task implements RunnableTask<Classification.Output> {

    @Schema(title = "Text prompt", description = "The input text to classify")
    @NotNull
    private Property<String> prompt;

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

        // Render input properties
        String renderedPrompt = runContext.render(prompt).as(String.class).orElseThrow();
        List<String> renderedClasses = runContext.render(classes).asList(String.class);

        // Get the appropriate model from the factory
        ChatModel model = this.provider.chatModel(runContext, configuration);

        String classificationPrompt = renderedPrompt +
            "\nRespond by only one of the following classes by typing just the exact class name: " + renderedClasses;

        // Perform text classification
        ChatResponse classificationResponse = model.chat(UserMessage.userMessage(classificationPrompt));
        logger.debug("Generated Classification: {}", classificationResponse.aiMessage().text());

        // send metrics for token usage
        TokenUsage tokenUsage = TokenUsage.from(classificationResponse.tokenUsage());
        AIUtils.sendMetrics(runContext, tokenUsage);

        return Output.builder()
            .classification(classificationResponse.aiMessage().text())
            .tokenUsage(tokenUsage)
            .finishReason(classificationResponse.finishReason())
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
