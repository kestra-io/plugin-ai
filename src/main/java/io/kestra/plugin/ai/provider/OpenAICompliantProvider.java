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
public abstract class OpenAICompliantProvider extends ModelProvider {
    @Schema(title = "API Key")
    @NotNull
    private Property<String> apiKey;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        if (configuration.getTopK() != null) {
            throw new IllegalArgumentException("OpenAI models do not support setting the topK parameter.");
        }

        OpenAiChatModel.OpenAiChatModelBuilder chatModelBuilder = OpenAiChatModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .baseUrl(runContext.render(getBaseUrl()).as(String.class).orElseThrow())
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .logRequests(runContext.render(configuration.getLogRequests()).as(Boolean.class).orElse(false))
            .seed(runContext.render(configuration.getSeed()).as(Integer.class).orElse(null))
            .logResponses(runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false))
            .logger(runContext.logger())
            .responseFormat(configuration.computeResponseFormat(runContext))
            .returnThinking(runContext.render(configuration.getReturnThinking()).as(Boolean.class).orElse(null))
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
            .baseUrl(runContext.render(getBaseUrl()).as(String.class).orElseThrow())
            .build();
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return OpenAiEmbeddingModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .baseUrl(runContext.render(getBaseUrl()).as(String.class).orElseThrow())
            .build();
    }
}
