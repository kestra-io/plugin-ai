package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.mistralai.MistralAiChatModel;
import dev.langchain4j.model.mistralai.MistralAiEmbeddingModel;
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

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(
    title = "Mistral AI Model Provider"
)
@Plugin(
    beta = true,
    examples = {
        @Example(
            title = "Chat completion with Mistral AI",
            full = true,
            code = {
                """
                id: chat_completion
                namespace: company.team

                inputs:
                  - id: prompt
                    type: STRING

                tasks:
                  - id: chat_completion
                    type: io.kestra.plugin.ai.ChatCompletion
                    provider:
                      type: io.kestra.plugin.ai.provider.MistralAI
                      apiKey: "{{ secret('MISTRAL_API_KEY') }}"
                      modelName: mistral:7b
                    messages:
                      - type: SYSTEM
                        content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                      - type: USER
                        content: "{{inputs.prompt}}"
                """
            }
        )
    },
    aliases = "io.kestra.plugin.langchain4j.provider.MistralAI"
)
public class MistralAI extends ModelProvider {
    @Schema(title = "API Key")
    @NotNull
    private Property<String> apiKey;

    @Schema(title = "API base URL")
    private Property<String> baseUrl;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        if (configuration.getTopK() != null) {
            throw new IllegalArgumentException("Mistral models didn't support topK");
        }

        return MistralAiChatModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .baseUrl(runContext.render(this.baseUrl).as(String.class).orElse(null))
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .randomSeed(runContext.render(configuration.getSeed()).as(Integer.class).orElse(null))
            .logRequests(true)
            .logRequests(runContext.render(configuration.getLogRequest()).as(Boolean.class).orElse(false))
            .logResponses(runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false))
            .build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) {
        throw new UnsupportedOperationException("MistralAI didn't support image model");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return MistralAiEmbeddingModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .baseUrl(runContext.render(this.baseUrl).as(String.class).orElse(null))
            .build();

    }

}
