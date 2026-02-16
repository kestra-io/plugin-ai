package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.ibm.watsonx.ai.CloudRegion;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.watsonx.WatsonxChatModel;
import dev.langchain4j.model.watsonx.WatsonxEmbeddingModel;
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
    title = "Use IBM watsonx.ai models",
    description = "Calls IBM watsonx.ai chat/embedding endpoints with API key and project ID. Ensure the selected model ID is available in the configured project."
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with Watsonx AI",
            full = true,
            code = """
                id: chat_completion
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING

                tasks:
                  - id: chat_completion
                    type: io.kestra.plugin.ai.completion.ChatCompletion
                    provider:
                      type: io.kestra.plugin.ai.provider.WatsonxAI
                      apiKey: "{{ kv('WATSONX_API_KEY') }}"
                      projectId: "{{ kv('WATSONX_PROJECT_ID') }}"
                      modelName: ibm/granite-3-3-8b-instruct
                      baseUrl : "https://api.eu-de.dataplatform.cloud.ibm.com/wx"
                    messages:
                      - type: SYSTEM
                        content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                      - type: USER
                        content: "{{ inputs.prompt }}"
                """
        )
    }
)
public class WatsonxAI extends ModelProvider {
    @Schema(title = "API Key")
    @NotNull
    private Property<String> apiKey;

    @Schema(title = "Project Id")
    @NotNull
    private Property<String> projectId;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        return WatsonxChatModel.builder()
            .baseUrl(runContext.render(this.baseUrl).as(String.class).orElse(null))
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .projectId(runContext.render(this.projectId).as(String.class).orElseThrow())
            .modelName(runContext.render(this.getModelName()).as(String.class).orElse("ibm/granite-3-3-8b-instruct"))
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(0.7))
            .maxOutputTokens(runContext.render(configuration.getMaxToken()).as(Integer.class).orElse(512))
            .build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        throw new UnsupportedOperationException("Watsonx is currently not supported for image generation.");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return WatsonxEmbeddingModel.builder()
            .baseUrl(runContext.render(this.baseUrl).as(String.class).orElse(null))
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .projectId(runContext.render(this.projectId).as(String.class).orElseThrow())
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .build();
    }
}
