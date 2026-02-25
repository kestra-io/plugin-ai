package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.http.client.jdk.JdkHttpClientBuilder;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiImageModel;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.util.List;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(
    title = "Use OpenRouter models",
    description = "Routes requests through OpenRouter’s multi-model API using your API key. Supports chat/embedding/image models exposed by OpenRouter; honor provider-specific safety/usage limits."
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with OpenRouter",
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
                          type: io.kestra.plugin.ai.provider.OpenRouter
                          apiKey: "{{ secret('OPENROUTER_API_KEY') }}"
                          baseUrl: https://openrouter.ai/api/v1
                          modelName: x-ai/grok-beta
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                          - type: USER
                            content: "{{inputs.prompt}}"
                    """
            }
        )
    },
    aliases = "io.kestra.plugin.langchain4j.provider.OpenRouter"
)
public class OpenRouter extends ModelProvider {

    private static final String BASE_URL = "https://openrouter.ai/api/v1";

    @Schema(title = "API Key")
    @NotNull
    private Property<String> apiKey;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        OpenAiChatModel.OpenAiChatModelBuilder chatModelBuilder = OpenAiChatModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .baseUrl(runContext.render(baseUrl).as(String.class).orElse(BASE_URL))
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .seed(runContext.render(configuration.getSeed()).as(Integer.class).orElse(null))
            .logRequests(runContext.render(configuration.getLogRequests()).as(Boolean.class).orElse(false))
            .logResponses(runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false))
            .logger(runContext.logger())
            .responseFormat(configuration.computeResponseFormat(runContext))
            .listeners(List.of(new TimingChatModelListener()))
            .maxCompletionTokens(runContext.render(configuration.getMaxToken()).as(Integer.class).orElse(null));

        JdkHttpClientBuilder httpClientBuilder = buildHttpClientWithPemIfAvailable(runContext);
        if (httpClientBuilder != null) {
            chatModelBuilder.httpClientBuilder(httpClientBuilder);
        }

        return chatModelBuilder.build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return OpenAiImageModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .baseUrl(runContext.render(baseUrl).as(String.class).orElse(BASE_URL))
            .build();
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return OpenAiEmbeddingModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .baseUrl(runContext.render(baseUrl).as(String.class).orElse(BASE_URL))
            .build();
    }

}
