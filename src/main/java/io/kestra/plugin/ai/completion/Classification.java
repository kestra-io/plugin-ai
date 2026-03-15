package io.kestra.plugin.ai.completion;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;

import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Metric;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.ListUtils;
import io.kestra.plugin.ai.AIUtils;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.GuardrailRule;
import io.kestra.plugin.ai.domain.Guardrails;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.ai.domain.TokenUsage;

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

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Classify text into provided classes",
    description = """
        Uses an LLM to assign the prompt to exactly one category from `classes`. A default system prompt forces a single-label reply; override it if you need different behavior. Output includes token usage and finish reason."""
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

    @Schema(title = "Text prompt", description = "The input text to classify")
    @NotNull
    private Property<String> prompt;

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

        String rPrompt = runContext.render(prompt).as(String.class).orElseThrow();
        List<String> rClasses = runContext.render(classes).asList(String.class);
        String rSystemMessage = runContext.render(systemMessage).as(String.class, Map.of("classes", rClasses)).orElseThrow();

        // Input guardrail check
        String inputViolation = evaluateGuardrails(ListUtils.emptyOnNull(guardrails != null ? guardrails.getInput() : null), Map.of("message", rPrompt), runContext);
        if (inputViolation != null) {
            logger.warn("Input guardrail violated: {}", inputViolation);
            return Output.builder().guardrailViolated(true).guardrailViolationMessage("Input guardrail: " + inputViolation).build();
        }

        List<dev.langchain4j.data.message.ChatMessage> chatMessages = new ArrayList<>();
        chatMessages.add(SystemMessage.systemMessage(rSystemMessage));
        chatMessages.add(UserMessage.userMessage(rPrompt));

        Duration taskTimeout = runContext.render(this.getTimeout()).as(Duration.class).orElse(Duration.ofSeconds(120));
        ChatModel model = this.provider.chatModel(runContext, configuration, taskTimeout);
        ChatResponse response = model.chat(chatMessages);

        logger.debug("Generated Classification: {}", response.aiMessage().text());

        // Output guardrail check
        String outputViolation = evaluateGuardrails(ListUtils.emptyOnNull(guardrails != null ? guardrails.getOutput() : null), Map.of("response", response.aiMessage().text()), runContext);
        if (outputViolation != null) {
            logger.warn("Output guardrail violated: {}", outputViolation);
            return Output.builder().guardrailViolated(true).guardrailViolationMessage("Output guardrail: " + outputViolation).build();
        }

        TokenUsage tokenUsage = TokenUsage.from(response.tokenUsage());
        AIUtils.sendMetrics(runContext, tokenUsage);

        return Output.builder()
            .classification(response.aiMessage().text())
            .tokenUsage(tokenUsage)
            .finishReason(response.finishReason())
            .build();
    }

    private String evaluateGuardrails(List<GuardrailRule> rules, Map<String, Object> context, RunContext runContext) {
        for (GuardrailRule rule : rules) {
            try {
                String result = runContext.render(Property.<String> ofExpression(rule.getExpression()))
                    .as(String.class, context).orElse("false");
                if (!Boolean.parseBoolean(result.trim())) {
                    return rule.getMessage();
                }
            } catch (Exception e) {
                return "Guardrail expression evaluation failed: " + e.getMessage();
            }
        }
        return null;
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
