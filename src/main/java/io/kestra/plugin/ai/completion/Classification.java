package io.kestra.plugin.ai.completion;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
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
        Uses an LLM to assign the input (`prompt` or `contentBlocks`) to exactly one category from `classes`. A default system prompt forces a single-label reply; override it if you need different behavior. Output includes token usage and finish reason."""
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
                          modelName: gemini-3.5-flash-lite
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
    aliases = { "io.kestra.plugin.langchain4j.Classification", "io.kestra.plugin.langchain4j.completion.Classification" }
)
public class Classification extends Task implements RunnableTask<Classification.Output> {

    @Schema(
        title = "Text prompt",
        description = "Text to classify. Set either this property or `contentBlocks`, not both.",
        example = "{{ inputs.ticket_body }}"
    )
    @Nullable
    @PluginProperty(group = "main")
    private Property<String> prompt;

    @Schema(
        title = "Content blocks",
        description = "Multimodal input to classify, as a list of `TEXT`, `IMAGE` or `PDF` blocks. Set either this property or `prompt`, not both. For `IMAGE` and `PDF` blocks, the `uri` supports the `kestra://`, `file://` and `nsfile://` schemes.",
        example = "[{type: \"TEXT\", text: \"Classify this invoice\"}, {type: \"PDF\", uri: \"{{ inputs.file }}\"}]"
    )
    @Nullable
    private Property<List<ChatMessage.ContentBlock>> contentBlocks;

    @Schema(
        title = "System message",
        description = "Instruction steering how the model classifies the input. Defaults to `Respond by only one of the following classes by typing just the exact class name: {{ classes }}`.",
        example = "Respond by only one of the following classes by typing just the exact class name: {{ classes }}"
    )
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<String> systemMessage = Property.ofExpression(
        "Respond by only one of the following classes by typing just the exact class name: {{ classes }}"
    );

    @Schema(
        title = "Classification options",
        description = "Categories the model must choose from, one of which is returned as the classification. No default: this property is required.",
        example = "[\"ACCOUNT\", \"BILLING\", \"TECHNICAL\", \"GENERAL\"]"
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<List<String>> classes;

    @Schema(
        title = "Language model provider",
        description = "Model provider that performs the classification. No default: this property is required.",
        example = "{type: \"io.kestra.plugin.ai.provider.GoogleGemini\", apiKey: \"{{ secret('GEMINI_API_KEY') }}\", modelName: \"gemini-3.5-flash-lite\"}"
    )
    @NotNull
    @PluginProperty(group = "main")
    private ModelProvider provider;

    @Schema(
        title = "Chat configuration",
        description = "Chat model settings (temperature, token limits, and so on). Defaults to an empty configuration, so the provider's own defaults apply; a low temperature gives the most consistent classifications.",
        example = "{temperature: 0.1}"
    )
    @NotNull
    @PluginProperty(group = "advanced")
    @Builder.Default
    private ChatConfiguration configuration = ChatConfiguration.empty();

    @Schema(
        title = "Guardrails",
        description = "Rules validating the call: input guardrails run against the prompt before the LLM is called, output guardrails against the classification before it is returned. The first failing rule stops execution and sets `guardrailViolated` to `true` in the output. Not set by default.",
        example = "{input: [{type: \"io.kestra.plugin.ai.guardrail.ExpressionInputGuardrail\", expression: \"{{ prompt | length < 5000 }}\"}]}"
    )
    @PluginProperty(group = "advanced")
    @Nullable
    private Guardrails guardrails;

    @Override
    public Classification.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        String rPrompt = prompt == null ? null : runContext.render(prompt).as(String.class).orElse(null);
        List<ChatMessage.ContentBlock> rContentBlocks = contentBlocks == null ? null : runContext.render(contentBlocks).asList(ChatMessage.ContentBlock.class);
        CompletionInputContentUtils.validatePromptInput("Classification", rPrompt, rContentBlocks);
        List<String> rClasses = runContext.render(classes).asList(String.class);
        String rSystemMessage = runContext.render(systemMessage).as(String.class, Map.of("classes", rClasses)).orElseThrow();

        // Input guardrail check
        String inputViolation = GuardrailsEvaluator.checkInput(guardrails, rPrompt, runContext);
        if (inputViolation != null) {
            return buildGuardrailViolationOutput(logger, "Input guardrail violated: {}", inputViolation, "Input guardrail: ");
        }

        List<dev.langchain4j.data.message.ChatMessage> chatMessages = new ArrayList<>();
        chatMessages.add(SystemMessage.systemMessage(rSystemMessage));
        chatMessages.add(CompletionInputContentUtils.toUserMessage(runContext, rPrompt, rContentBlocks));

        Duration taskTimeout = runContext.render(this.getTimeout()).as(Duration.class).orElse(Duration.ofSeconds(120));
        ChatModel model = TokenBudgetChatModel.wrap(
            this.provider.chatModel(runContext, configuration, taskTimeout),
            runContext,
            configuration
        );
        runContext.metric(Counter.of("ai.provider.calls", 1, "provider", this.provider.getClass().getName()));
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
            title = "Classification result",
            description = "Category the model assigned to the input, taken from `classes`.",
            example = "BILLING"
        )
        private final String classification;

        @Schema(
            title = "Token usage",
            description = "Input, output and total tokens billed for the call, when the provider reports them.",
            example = "{inputTokenCount: 42, outputTokenCount: 3, totalTokenCount: 45}"
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
}
