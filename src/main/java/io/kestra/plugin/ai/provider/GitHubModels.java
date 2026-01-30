package io.kestra.plugin.ai.provider;

import com.azure.ai.inference.models.ChatCompletionsResponseFormat;
import com.azure.ai.inference.models.ChatCompletionsResponseFormatJsonObject;
import com.azure.ai.inference.models.ChatCompletionsResponseFormatText;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.github.GitHubModelsChatModel;
import dev.langchain4j.model.github.GitHubModelsEmbeddingModel;
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
    title = "Use GitHub Models via Azure AI Inference",
    description = "Calls GitHub Models through the Azure AI Inference API with a GitHub token. Supports chat and embeddings; response format options map to Azure AI Inference capabilities."
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with GitHub Models",
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
                          type: io.kestra.plugin.ai.provider.GitHubModels
                          gitHubToken: "{{ kv('GITHUB_TOKEN') }}"
                          modelName: gpt-4o-mini
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely.
                          - type: USER
                            content: "{{inputs.prompt}}"
                    """
            }
        )
    }
)
public class GitHubModels extends ModelProvider {

    @Schema(
        title = "GitHub Token",
        description = "Personal Access Token (PAT) used to access GitHub Models."
    )
    @NotNull
    private Property<String> gitHubToken;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration)
        throws IllegalVariableEvaluationException {

        var logRequests = runContext.render(configuration.getLogRequests()).as(Boolean.class).orElse(false);
        var logResponses = runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false);
        var seed = runContext.render(configuration.getSeed()).as(Integer.class).orElse(null);

        var builder = GitHubModelsChatModel.builder()
            .gitHubToken(runContext.render(this.gitHubToken).as(String.class).orElseThrow())
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .seed(seed != null ? seed.longValue() : null)
            .responseFormat(toAzureResponseFormat(runContext, configuration))
            .maxTokens(runContext.render(configuration.getMaxToken()).as(Integer.class).orElse(null))
            .logRequestsAndResponses(logRequests || logResponses)
            .listeners(List.of(new TimingChatModelListener()));

        runContext.render(this.baseUrl).as(String.class).ifPresent(builder::endpoint);

        return builder.build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) {
        throw new UnsupportedOperationException("GitHub Models is not supported for image generation.");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext)
        throws IllegalVariableEvaluationException {

        var logRequests = runContext.render(Property.ofValue(false)).as(Boolean.class).orElse(false);
        var logResponses = runContext.render(Property.ofValue(false)).as(Boolean.class).orElse(false);

        var builder = GitHubModelsEmbeddingModel.builder()
            .gitHubToken(runContext.render(this.gitHubToken).as(String.class).orElseThrow())
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .logRequestsAndResponses(logRequests || logResponses);

        runContext.render(this.baseUrl).as(String.class).ifPresent(builder::endpoint);

        return builder.build();
    }

    private ChatCompletionsResponseFormat toAzureResponseFormat(
        RunContext runContext,
        ChatConfiguration configuration
    ) throws IllegalVariableEvaluationException {

        var lc4jFormat = configuration.computeResponseFormat(runContext);

        if (lc4jFormat.jsonSchema() != null) {
            runContext.logger().warn(
                "GitHub Models responseFormat supports JSON mode but does not enforce JSON Schema. " +
                    "Use prompt constraints and validate downstream."
            );
        }

        if (lc4jFormat.type() == dev.langchain4j.model.chat.request.ResponseFormatType.JSON) {
            return new ChatCompletionsResponseFormatJsonObject();
        }

        return new ChatCompletionsResponseFormatText();
    }
}
