package io.kestra.plugin.ai.provider;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;

import dev.langchain4j.http.client.jdk.JdkHttpClientBuilder;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GeminiThinkingConfig;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.image.ImageModel;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
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
    title = "Use Google Gemini models",
    description = """
        Supports Gemini chat, embeddings, and images. Tools do not support JSON Schema `anyOf`, and tools cannot be combined with responseFormat; configure either but not both."""
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with Google Gemini",
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
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          apiKey: "{{ secret('GOOGLE_API_KEY') }}"
                          modelName: gemini-2.5-flash
                          thinkingEnabled: true
                          thinkingBudgetTokens: 1024
                          returnThinking: true
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            }
        ),
        @Example(
            title = "Chat completion with Google Gemini with a local base URL + PEM certificates",
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
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-2.5-flash
                          clientPem: "{{ secret('CLIENT_PEM') }}"
                          caPem: "{{ secret('CA_PEM') }}"
                          baseUrl: "https://internal.gemini.company.com/endpoint"
                          thinkingEnabled: true
                          thinkingBudgetTokens: 1024
                          returnThinking: true
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            }
        )
    },
    aliases = "io.kestra.plugin.langchain4j.provider.GoogleGemini"
)
public class GoogleGemini extends ModelProvider {

    @Schema(
        title = "API Key",
        description = "Required unless certificate-based authentication is configured with `clientPem` (optionally with `caPem`)."
    )
    @PluginProperty(group = "connection")
    private Property<String> apiKey;

    @Schema(title = "The configuration for embeddingModel")
    @PluginProperty(group = "advanced")
    private EmbeddingModelConfiguration embeddingModelConfiguration;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        return chatModel(runContext, configuration, Duration.ofSeconds(120), Collections.emptyList());
    }

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout, List<ChatModelListener> additionalListeners) throws IllegalVariableEvaluationException {
        var allListeners = new ArrayList<ChatModelListener>();
        allListeners.add(new TimingChatModelListener());
        allListeners.addAll(additionalListeners);

        var rApiKey = resolveAuth(runContext);
        GoogleAiGeminiChatModel.GoogleAiGeminiChatModelBuilder chatModelBuilder = GoogleAiGeminiChatModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
            .topK(runContext.render(configuration.getTopK()).as(Integer.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .seed(runContext.render(configuration.getSeed()).as(Integer.class).orElse(null))
            .logRequests(runContext.render(configuration.getLogRequests()).as(Boolean.class).orElse(false))
            .logResponses(runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false))
            .logger(runContext.logger())
            .responseFormat(configuration.computeResponseFormat(runContext))
            .listeners(allListeners)
            .thinkingConfig(getThinkingConfig(configuration, runContext))
            .returnThinking(runContext.render(configuration.getReturnThinking()).as(Boolean.class).orElse(null))
            .maxOutputTokens(runContext.render(configuration.getMaxToken()).as(Integer.class).orElse(null))
            .timeout(timeout);

        if (rApiKey != null) {
            chatModelBuilder.apiKey(rApiKey);
        }

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
        throw new UnsupportedOperationException("Gemini is currently not supported for image generation.");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        var rApiKey = resolveAuth(runContext);
        GoogleAiEmbeddingModel.GoogleAiEmbeddingModelBuilder builder = GoogleAiEmbeddingModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .logger(runContext.logger());

        if (rApiKey != null) {
            builder.apiKey(rApiKey);
        }

        if (embeddingModelConfiguration != null) {
            builder
                .titleMetadataKey(runContext.render(this.embeddingModelConfiguration.titleMetadataKey).as(String.class).orElse(null))
                .taskType(runContext.render(this.embeddingModelConfiguration.taskType).as(GoogleAiEmbeddingModel.TaskType.class).orElse(null))
                .maxRetries(runContext.render(this.embeddingModelConfiguration.maxRetries).as(Integer.class).orElse(null))
                .outputDimensionality(runContext.render(this.embeddingModelConfiguration.outputDimensionality).as(Integer.class).orElse(null))
                .timeout(runContext.render(this.embeddingModelConfiguration.timeout).as(Duration.class).orElse(null));
        }

        return builder.build();
    }

    String resolveAuth(RunContext runContext) throws IllegalVariableEvaluationException {
        var rApiKey = runContext.render(this.apiKey).as(String.class).orElse(null);
        var rClientPem = runContext.render(this.getClientPem()).as(String.class).orElse(null);

        if (rApiKey == null && rClientPem == null) {
            throw new IllegalArgumentException(
                "GoogleGemini requires either `apiKey` or `clientPem` (optionally with `caPem`) for certificate-based authentication."
            );
        }

        return rApiKey;
    }

    private static GeminiThinkingConfig getThinkingConfig(final ChatConfiguration configuration, final RunContext runContext) throws IllegalVariableEvaluationException {
        var enabled = runContext.render(configuration.getThinkingEnabled()).as(Boolean.class).orElse(false);
        var maxTokens = runContext.render(configuration.getThinkingBudgetTokens()).as(Integer.class).orElse(null);
        return GeminiThinkingConfig.builder()
            .includeThoughts(enabled)
            .thinkingBudget(maxTokens)
            .build();
    }

    @Getter
    @Builder
    private static class EmbeddingModelConfiguration {
        @Schema(
            title = "The headline or name of the document (passed to the model as metadata).",
            description = "If set, this help improving retrieval quality by providing context for a document."
        )
        @PluginProperty(group = "connection")
        private Property<String> titleMetadataKey;

        @Schema(title = "Used to convey intended downstream application to help the model produce better embeddings.")
        private Property<GoogleAiEmbeddingModel.TaskType> taskType;

        @Schema(title = "Maximum number of retries for failed requests")
        @PluginProperty(group = "execution")
        private Property<Integer> maxRetries;

        @Schema(title = "Timeout in seconds for each request")
        @PluginProperty(group = "execution")
        private Property<Duration> timeout;

        @Schema(title = "Used to specify output embedding size", description = "If set, output embeddings will be truncated to the size specified.")
        @PluginProperty(group = "advanced")
        private Property<Integer> outputDimensionality;
    }
}
