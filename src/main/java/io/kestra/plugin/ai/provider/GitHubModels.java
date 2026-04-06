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

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
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
                          gitHubToken: "{{ secret('GITHUB_TOKEN') }}"
                          modelName: gpt-4o-mini
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely.
                          - type: USER
                            content: "{{ inputs.prompt }}"
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
    @PluginProperty(group = "main")
    private Property<String> gitHubToken;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration)
        throws IllegalVariableEvaluationException {
        return chatModel(runContext, configuration, Duration.ofSeconds(120), Collections.emptyList());
    }

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout, List<ChatModelListener> additionalListeners)
        throws IllegalVariableEvaluationException {

        var allListeners = new ArrayList<ChatModelListener>();
        allListeners.add(new TimingChatModelListener());
        allListeners.addAll(additionalListeners);

        var logRequests = runContext.render(configuration.getLogRequests()).as(Boolean.class).orElse(false);
        var logResponses = runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false);
        var seed = runContext.render(configuration.getSeed()).as(Integer.class).orElse(null);

        var builder = OpenAiChatModel.builder()
            .apiKey(runContext.render(this.gitHubToken).as(String.class).orElseThrow())
            .baseUrl(runContext.render(this.baseUrl).as(String.class).orElse("https://models.inference.ai.azure.com"))
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .seed(seed)
            .responseFormat(toAzureResponseFormat(runContext, configuration))
            .maxCompletionTokens(runContext.render(configuration.getMaxToken()).as(Integer.class).orElse(null))
            .logRequests(logRequests)
            .logResponses(logResponses)
            .listeners(allListeners)
            .timeout(timeout);

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

        var builder = OpenAiEmbeddingModel.builder()
            .apiKey(runContext.render(this.gitHubToken).as(String.class).orElseThrow())
            .baseUrl(runContext.render(this.baseUrl).as(String.class).orElse("https://models.inference.ai.azure.com"))
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .logRequests(logRequests)
            .logResponses(logResponses);

        return builder.build();
    }

    private ResponseFormat toAzureResponseFormat(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        var lc4jFormat = configuration.computeResponseFormat(runContext);

        if (lc4jFormat.jsonSchema() != null) {
            runContext.logger().warn(
                "GitHub Models responseFormat supports JSON mode but does not enforce JSON Schema. " +
                    "Use prompt constraints and validate downstream."
            );
        }

        if (lc4jFormat.type() == ResponseFormatType.JSON) {
            return ResponseFormat.builder().type(ResponseFormatType.JSON).build();
        }

        return ResponseFormat.TEXT;
    }
}
