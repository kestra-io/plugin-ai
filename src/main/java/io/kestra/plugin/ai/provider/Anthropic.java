package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.model.anthropic.AnthropicChatModel;
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
    title = "Anthropic AI Model Provider"
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
                    type: io.kestra.plugin.ai.ChatCompletion
                    provider:
                      type: io.kestra.plugin.ai.provider.Anthropic
                      apiKey: "{{ kv('ANTHROPIC_API_KEY') }}"
                      modelName: claude-3-haiku-20240307
                    messages:
                      - type: SYSTEM
                        content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                      - type: USER
                        content: "{{inputs.prompt}}"
                """
            }
        )
    },
    aliases = "io.kestra.plugin.langchain4j.provider.Anthropic"
)
public class Anthropic extends ModelProvider {
    @Schema(title = "API Key")
    @NotNull
    private Property<String> apiKey;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        if (configuration.getSeed() != null) {
            throw new IllegalArgumentException("Anthropic models didn't support setting the seed");
        }

        if (configuration.getResponseFormat() != null) {
            throw new IllegalVariableEvaluationException("Anthropic models didn't support configuring the response format");
        }

        return AnthropicChatModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
            .topK(runContext.render(configuration.getTopK()).as(Integer.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .logRequests(runContext.render(configuration.getLogRequests()).as(Boolean.class).orElse(false))
            .logResponses(runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false))
            .listeners(List.of(new TimingChatModelListener()))
            .build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) {
        throw new UnsupportedOperationException("Anthropic didn't support image model");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        throw new UnsupportedOperationException("Anthropic doesn’t support embeddings");
    }

}
