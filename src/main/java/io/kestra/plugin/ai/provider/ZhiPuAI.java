package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.community.model.zhipu.ZhipuAiChatModel;
import dev.langchain4j.community.model.zhipu.ZhipuAiEmbeddingModel;
import dev.langchain4j.community.model.zhipu.ZhipuAiImageModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;

import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(title = "ZhiPu AI Model Provider")
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with ZhiPu AI",
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
                        type: io.kestra.plugin.ai.ChatCompletion
                        provider:
                          type: io.kestra.plugin.ai.provider.ZhiPuAI
                          apiKey: "{{ kv('ZHIPU_API_KEY') }}"
                          modelName: glm-4.5-flash
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            })
    })
public class ZhiPuAI extends ModelProvider {
    private static final String BASE_URL = "https://open.bigmodel.cn/";

    @Schema(title = "API Key")
    @NotNull
    private Property<String> apiKey;

    @Schema(
        title = "API base URL", description = "The base URL for ZhiPu API (defaults to https://open.bigmodel.cn/)")
    @NotNull
    @Builder.Default
    private Property<String> baseUrl = Property.ofValue(BASE_URL);

    @Schema(title = "With the stop parameter, the model will automatically stop generating text when it is about to contain the specified string or token_id")
    private Property<List<String>> stops;

    @Schema(title = "The maximum retry times to request")
    private Property<Integer> maxRetries;

    @Schema(title = "The maximum number of tokens returned by this request")
    private Property<Integer> maxToken;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        if (configuration.getTopK() != null) {
            throw new IllegalArgumentException("ZhiPu AI models do not support setting the topK parameter.");
        }

        if (configuration.getSeed() != null) {
            throw new IllegalArgumentException("ZhiPu AI  models do not support setting the seed.");
        }

        if (configuration.getResponseFormat() != null) {
            throw new IllegalVariableEvaluationException("ZhiPu AI  models do not support configuring the response format.");
        }

        return ZhipuAiChatModel.builder()
            .baseUrl(runContext.render(baseUrl).as(String.class).orElse(BASE_URL))
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .model(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .logRequests(runContext.render(configuration.getLogRequests()).as(Boolean.class).orElse(false))
            .logResponses(runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false))
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(0.7))
            .stops(runContext.render(this.stops).asList(String.class))
            .maxRetries(runContext.render(this.maxRetries).as(Integer.class).orElse(3))
            .maxToken(runContext.render(this.maxToken).as(Integer.class).orElse(512))
            .listeners(List.of(new TimingChatModelListener()))
            .build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return ZhipuAiImageModel.builder()
            .model(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .baseUrl(runContext.render(baseUrl).as(String.class).orElse(null))
            .build();
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return ZhipuAiEmbeddingModel.builder()
            .model(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .baseUrl(runContext.render(baseUrl).as(String.class).orElse(null))
            .build();
    }
}
