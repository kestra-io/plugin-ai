package io.kestra.plugin.langchain4j.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.langchain4j.domain.ModelProvider;
import io.kestra.plugin.langchain4j.domain.ChatConfiguration;
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
@Plugin(beta = true)
@JsonDeserialize
@Schema(
    title = "Ollama Model Provider"
)
public class Ollama extends ModelProvider {
    @Schema(title = "Model endpoint")
    @NotNull
    private Property<String> endpoint;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        return OllamaChatModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .baseUrl(runContext.render(this.endpoint).as(String.class).orElseThrow())
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
            .topK(runContext.render(configuration.getTopK()).as(Integer.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .seed(runContext.render(configuration.getSeed()).as(Integer.class).orElse(null))
            .build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) {
        throw new UnsupportedOperationException("Ollama didn't support image generation");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return OllamaEmbeddingModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .baseUrl(runContext.render(this.endpoint).as(String.class).orElseThrow())
            .build();
    }
}
