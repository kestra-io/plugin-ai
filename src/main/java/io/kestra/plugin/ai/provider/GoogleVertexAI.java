package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.model.vertexai.VertexAiImageModel;
import dev.langchain4j.model.vertexai.gemini.SchemaHelper;
import dev.langchain4j.model.vertexai.gemini.VertexAiGeminiChatModel;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.ai.domain.ChatConfiguration;
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
    title = "Google VertexAI Model Provider"
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with Google Vertex AI",
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
                      type: io.kestra.plugin.ai.provider.GoogleVertexAI
                      endpoint: your-vertex-ai-endpoint
                      location: your-google-cloud-region
                      project: your-google-cloud-project-id
                    messages:
                      - type: SYSTEM
                        content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                      - type: USER
                        content: "{{inputs.prompt}}"
                """
            }
        )
    },
    aliases = "io.kestra.plugin.langchain4j.provider.GoogleVertexAI"
)
public class GoogleVertexAI extends ModelProvider {
    @Schema(title = "Endpoint URL")
    @NotNull
    private Property<String> endpoint;

    @Schema(title = "Project location")
    @NotNull
    private Property<String> location;

    @Schema(title = "Project ID")
    @NotNull
    private Property<String> project;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        if (this.endpoint != null) {
            throw new IllegalArgumentException("The `endpoint` property cannot be used for the Chat Model which uses Gemini only.");
        }

        var responseFormat = configuration.computeResponseFormat(runContext);

        return VertexAiGeminiChatModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .location(runContext.render(this.location).as(String.class).orElseThrow())
            .project(runContext.render(this.project).as(String.class).orElseThrow())
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).map(d -> d.floatValue()).orElse(null))
            .topK(runContext.render(configuration.getTopK()).as(Integer.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).map(d -> d.floatValue()).orElse(null))
            .seed(runContext.render(configuration.getSeed()).as(Integer.class).orElse(null))
            .logRequests(runContext.render(configuration.getLogRequests()).as(Boolean.class).orElse(false))
            .logResponses(runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false))
            .responseMimeType(responseFormat.type() == ResponseFormatType.JSON ? "application/json" : null)
            .responseSchema(responseFormat.jsonSchema() != null ? SchemaHelper.from(responseFormat.jsonSchema().rootElement()) : null)
            .listeners(List.of(new TimingChatModelListener()))
            .build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return VertexAiImageModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .endpoint(runContext.render(this.endpoint).as(String.class).orElse(null))
            .location(runContext.render(this.location).as(String.class).orElse(null))
            .project(runContext.render(this.project).as(String.class).orElseThrow())
            .build();
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return VertexAiEmbeddingModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .endpoint(runContext.render(this.endpoint).as(String.class).orElse(null))
            .location(runContext.render(this.location).as(String.class).orElse(null))
            .project(runContext.render(this.project).as(String.class).orElseThrow())
            .build();
    }
}
