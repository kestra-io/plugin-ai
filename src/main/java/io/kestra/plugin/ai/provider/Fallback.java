package io.kestra.plugin.ai.provider;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Configure ordered model providers for fallback",
    description = "Tries chat providers in configured order, falling back on connection failures, timeouts, and retriable provider errors such as HTTP 5xx and 429. Each provider attempt receives the full configured timeout independently, so multiple timeouts can make total duration approach N times that timeout, plus overhead; this is not guaranteed because providers differ in how they apply timeouts. It does not fall back on authentication, invalid request, guardrail, tool execution, or JSON/response-processing failures. Image and embedding operations are unsupported."
)
@Plugin(
    examples = {
        @Example(
            title = "AI agent with ordered provider fallback",
            full = true,
            code = {
                """
                    id: ai_agent_with_fallback
                    namespace: company.ai

                    inputs:
                      - id: prompt
                        type: STRING

                    tasks:
                      - id: ai_agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        provider:
                          type: io.kestra.plugin.ai.provider.Fallback
                          providers:
                            - type: io.kestra.plugin.ai.provider.OpenAI
                              modelName: gpt-5-mini
                              apiKey: "{{ secret('OPENAI_API_KEY') }}"
                              baseUrl: https://api.openai.com/v1
                            - type: io.kestra.plugin.ai.provider.GoogleGemini
                              modelName: gemini-2.5-flash
                              apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        systemMessage: You are a helpful assistant. Answer clearly and concisely.
                        prompt: "{{ inputs.prompt }}"
                    """
            }
        )
    }
)
public class Fallback extends ModelProvider {
    @Override
    protected boolean isModelNameRequired() {
        return false;
    }

    @Schema(
        title = "Providers",
        description = "Model providers in the order they should be considered for fallback."
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<@Size(min = 1) List<ModelProvider>> providers;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        return chatModel(runContext, provider -> provider.chatModel(runContext, configuration));
    }

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout) throws IllegalVariableEvaluationException {
        return chatModel(runContext, provider -> provider.chatModel(runContext, configuration, timeout));
    }

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout, List<ChatModelListener> additionalListeners)
        throws IllegalVariableEvaluationException {
        return chatModel(runContext, provider -> provider.chatModel(runContext, configuration, timeout, additionalListeners));
    }

    private ChatModel chatModel(RunContext runContext, ChatModelFactory chatModelFactory)
        throws IllegalVariableEvaluationException {
        validateNoOuterConnectionProperties();
        List<ChatModel> childModels = new ArrayList<>();
        for (ModelProvider provider : renderProviders(runContext)) {
            childModels.add(chatModelFactory.create(provider));
        }
        return new FallbackChatModel(childModels, runContext.logger());
    }

    @FunctionalInterface
    private interface ChatModelFactory {
        ChatModel create(ModelProvider provider) throws IllegalVariableEvaluationException;
    }

    private List<ModelProvider> renderProviders(RunContext runContext) throws IllegalVariableEvaluationException {
        return runContext.render(providers).asList(ModelProvider.class);
    }

    private void validateNoOuterConnectionProperties() {
        if (getBaseUrl() != null || getClientPem() != null || getCaPem() != null) {
            throw new IllegalArgumentException(
                "Fallback does not use outer `baseUrl`, `clientPem`, or `caPem`; configure these properties on the appropriate nested provider instead."
            );
        }
    }

    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        throw new UnsupportedOperationException("Fallback provider supports chat models only.");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        throw new UnsupportedOperationException("Fallback provider supports chat models only.");
    }
}
