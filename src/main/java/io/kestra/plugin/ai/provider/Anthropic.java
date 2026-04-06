package io.kestra.plugin.ai.provider;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;

import dev.langchain4j.http.client.jdk.JdkHttpClientBuilder;
import dev.langchain4j.model.anthropic.AnthropicChatModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import io.kestra.core.models.annotations.PluginProperty;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(
    title = "Use Anthropic Claude models",
    description = "Provides Claude chat models only; embeddings and images are unsupported. Enforces Anthropic rules (no seed/responseFormat). Thinking mode requires `max_tokens` > `thinking.budget_tokens`; set API key and optional base URL."
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with Anthropic",
            full = true,
            code = {
                """
                    id: chat_completion
                    namespace: company.ai

                    inputs:
                      - id: prompt
                        type: STRING

                    tasks:
                      - id: chat_completion
                        type: io.kestra.plugin.ai.completion.ChatCompletion
                        provider:
                          type: io.kestra.plugin.ai.provider.Anthropic
                          apiKey: "{{ secret('ANTHROPIC_API_KEY') }}"
                          modelName: claude-3-haiku-20240307
                          thinkingEnabled: true
                          thinkingBudgetTokens: 1024
                          returnThinking: false
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            }
        )
    },
    aliases = "io.kestra.plugin.langchain4j.provider.Anthropic"
)
public class Anthropic extends ModelProvider {
    private static final String ENABLED = "enabled";
    @Schema(title = "API Key")
    @NotNull
    @PluginProperty(group = "main")
    private Property<String> apiKey;
    @Schema(
        title = "Maximum Tokens",
        description = """
            Specifies the maximum number of tokens that the model is allowed to generate in its response."""
    )
    @PluginProperty(group = "execution")
    private Property<Integer> maxTokens;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        return chatModel(runContext, configuration, Duration.ofSeconds(120), Collections.emptyList());
    }

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout, List<ChatModelListener> additionalListeners) throws IllegalVariableEvaluationException {
        if (configuration.getSeed() != null) {
            throw new IllegalArgumentException("Anthropic models do not support setting the seed.");
        }

        if (configuration.getResponseFormat() != null) {
            throw new IllegalVariableEvaluationException("Anthropic models do not support configuring the response format.");
        }
        var thinkingEnabled = runContext.render(configuration.getThinkingEnabled()).as(Boolean.class).orElse(false);
        var thinkingBudgetTokens = runContext.render(configuration.getThinkingBudgetTokens()).as(Integer.class).orElse(null);
        var maxTokens = runContext.render(this.getMaxTokens()).as(Integer.class).orElse(null);
        if (isInvalidThinkingConfig(thinkingEnabled, thinkingBudgetTokens, maxTokens)) {
            throw new IllegalArgumentException(
                "`max_tokens` must be greater than `thinking.budget_tokens` for thinking-enabled Anthropic models."
            );
        }

        var allListeners = new ArrayList<ChatModelListener>();
        allListeners.add(new TimingChatModelListener());
        allListeners.addAll(additionalListeners);

        AnthropicChatModel.AnthropicChatModelBuilder chatModelBuilder = AnthropicChatModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
            .topK(runContext.render(configuration.getTopK()).as(Integer.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .logRequests(runContext.render(configuration.getLogRequests()).as(Boolean.class).orElse(false))
            .logResponses(runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false))
            .logger(runContext.logger())
            .listeners(allListeners)
            .maxTokens(maxTokens) // Anthropic max tokens
            .thinkingType(thinkingEnabled ? ENABLED : null)
            .thinkingBudgetTokens(thinkingBudgetTokens)
            .returnThinking(runContext.render(configuration.getReturnThinking()).as(Boolean.class).orElse(null))
            .maxTokens(runContext.render(configuration.getMaxToken()).as(Integer.class).orElse(null))
            .cacheSystemMessages(runContext.render(configuration.getPromptCaching()).as(Boolean.class).orElse(null))
            .cacheTools(runContext.render(configuration.getPromptCaching()).as(Boolean.class).orElse(null));

        JdkHttpClientBuilder httpClientBuilder = buildHttpClientWithPemIfAvailable(runContext);
        if (httpClientBuilder != null) {
            chatModelBuilder.httpClientBuilder(httpClientBuilder);
        }

        String rBaseUrl = runContext.render(this.baseUrl).as(String.class).orElse(null);
        if (rBaseUrl != null) {
            chatModelBuilder.baseUrl(rBaseUrl);
        }

        return chatModelBuilder.build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) {
        throw new UnsupportedOperationException("Anthropic is currently not supported for image generation.");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        throw new UnsupportedOperationException("Anthropic is currently not supported for embedding models.");
    }

    private boolean isInvalidThinkingConfig(final boolean thinkingEnabled, final Integer thinkingBudgetTokens,
        final Integer maxTokens) {
        return thinkingEnabled
            && Objects.nonNull(thinkingBudgetTokens)
            && Objects.nonNull(maxTokens)
            && thinkingBudgetTokens > maxTokens;
    }
}
