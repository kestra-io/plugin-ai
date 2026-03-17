package io.kestra.plugin.ai.completion;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.model.chat.ChatModel;
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
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.Guardrails;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.ai.domain.TokenUsage;
import io.kestra.plugin.ai.guardrail.GuardrailsEvaluator;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.annotation.Nullable;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;
import org.slf4j.Logger;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Classify text into provided classes",
    description = """
        Uses an LLM to assign the input (`prompt` or `promptContentBlocks`) to exactly one category from `classes`. A default system prompt forces a single-label reply; override it if you need different behavior. Output includes token usage and finish reason."""
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
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                          modelName: gemini-2.5-flash
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
    aliases = { "io.kestra.plugin.langchain4j.Classification", "io.kestra.plugin.langchain4j.completion.Classification" }
)
public class Classification extends Task implements RunnableTask<Classification.Output> {

    @Schema(title = "Text prompt", description = "Text input to classify. Use either `prompt` or `promptContentBlocks`.")
    private Property<String> prompt;

    @Schema(
        title = "Prompt content blocks",
        description = "Multimodal input blocks for classification (TEXT, IMAGE, PDF). Use either `prompt` or `promptContentBlocks`."
    )
    @Nullable
    private Property<List<ChatMessage.ContentBlock>> promptContentBlocks;

    @Schema(
        title = "Optional system message",
        description = "Instruction message for the model. Defaults to a standard classification instruction using the provided classes."
    )
    @Builder.Default
    private Property<String> systemMessage = Property.ofExpression(
        "Respond by only one of the following classes by typing just the exact class name: {{ classes }}"
    );

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

    @Schema(
        title = "Guardrails",
        description = """
            Input guardrails are evaluated against the prompt before the LLM is called.
            Output guardrails are evaluated against the classification result before it is returned.
            The first failing rule stops execution and sets `guardrailViolated` to `true` in the output."""
    )
    @PluginProperty
    @Nullable
    private Guardrails guardrails;

    @Override
    public Classification.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        String rPrompt = prompt == null ? null : runContext.render(prompt).as(String.class).orElse(null);
        List<ChatMessage.ContentBlock> rPromptContentBlocks = promptContentBlocks == null ? null : runContext.render(promptContentBlocks).asList(ChatMessage.ContentBlock.class);
        validatePromptInput(rPrompt, rPromptContentBlocks);
        List<String> rClasses = runContext.render(classes).asList(String.class);
        String rSystemMessage = runContext.render(systemMessage).as(String.class, Map.of("classes", rClasses)).orElseThrow();

        // Input guardrail check
        String inputViolation = GuardrailsEvaluator.checkInput(guardrails, rPrompt, runContext);
        if (inputViolation != null) {
            return buildGuardrailViolationOutput(logger, "Input guardrail violated: {}", inputViolation, "Input guardrail: ");
        }

        List<dev.langchain4j.data.message.ChatMessage> chatMessages = new ArrayList<>();
        chatMessages.add(SystemMessage.systemMessage(rSystemMessage));
        chatMessages.add(CompletionInputContentUtils.toUserMessage(runContext, rPrompt, rPromptContentBlocks));

        Duration taskTimeout = runContext.render(this.getTimeout()).as(Duration.class).orElse(Duration.ofSeconds(120));
        ChatModel model = this.provider.chatModel(runContext, configuration, taskTimeout);
        ChatResponse response = model.chat(chatMessages);

        logger.debug("Generated Classification: {}", response.aiMessage().text());

        // Output guardrail check
        String outputViolation = GuardrailsEvaluator.checkOutput(guardrails, response, runContext);
        if (outputViolation != null) {
            return buildGuardrailViolationOutput(logger, "Output guardrail violated: {}", outputViolation, "Output guardrail: ");
        }

        TokenUsage tokenUsage = TokenUsage.from(response.tokenUsage());
        AIUtils.sendMetrics(runContext, tokenUsage);

        return Output.builder()
            .classification(response.aiMessage().text())
            .tokenUsage(tokenUsage)
            .finishReason(response.finishReason())
            .build();
    }

    private void validatePromptInput(String prompt, List<ChatMessage.ContentBlock> promptContentBlocks) {
        boolean hasPrompt = prompt != null && !prompt.isBlank();
        boolean hasPromptContentBlocks = promptContentBlocks != null && !promptContentBlocks.isEmpty();
        if (hasPrompt && hasPromptContentBlocks) {
            throw new IllegalArgumentException("Classification accepts either `prompt` or `promptContentBlocks`, but not both.");
        }
        if (!hasPrompt && !hasPromptContentBlocks) {
            throw new IllegalArgumentException("Classification requires one input source: `prompt` or `promptContentBlocks`.");
        }
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
        @Schema(title = "Classification Result", description = "The classified category of the input text")
        private final String classification;

        @Schema(title = "Token usage")
        private TokenUsage tokenUsage;

        @Schema(title = "Finish reason")
        private FinishReason finishReason;

        @Schema(title = "Guardrail violated", description = "True if a guardrail rule was violated")
        @Builder.Default
        private boolean guardrailViolated = false;

        @Schema(title = "Guardrail violation message", description = "The message from the first violated guardrail rule")
        private String guardrailViolationMessage;
    }
}
