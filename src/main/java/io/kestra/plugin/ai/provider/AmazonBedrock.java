package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.model.bedrock.BedrockChatModel;
import dev.langchain4j.model.bedrock.BedrockChatRequestParameters;
import dev.langchain4j.model.bedrock.BedrockCohereEmbeddingModel;
import dev.langchain4j.model.bedrock.BedrockTitanEmbeddingModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
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
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.services.bedrockruntime.BedrockRuntimeClient;

import java.util.List;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(
    title = "Amazon Bedrock Model Provider"
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with OpenAI",
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
                      type: io.kestra.plugin.ai.provider.AmazonBedrock
                      accessKeyId: "{{ kv('AWS_ACCESS_KEY') }}"
                      secretAccessKey: "{{ kv('AWS_SECRET_KEY') }}"
                      modelName: anthropic.claude-3-sonnet-20240229-v1:0
                    messages:
                      - type: SYSTEM
                        content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                      - type: USER
                        content: "{{inputs.prompt}}"
                """
            }
        )
    },
    aliases = "io.kestra.plugin.langchain4j.provider.AmazonBedrock"
)
public class AmazonBedrock extends ModelProvider {

    @Schema(title = "AWS Access Key ID")
    @NotNull
    private Property<String> accessKeyId;

    @Schema(title = "AWS Secret Access Key")
    @NotNull
    private Property<String> secretAccessKey;

    @Schema(title = "Amazon Bedrock Embedding Model Type")
    @NotNull
    @Builder.Default
    private Property<AmazonBedrockEmbeddingModelType> modelType = Property.ofValue(AmazonBedrockEmbeddingModelType.COHERE);

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        if (configuration.getSeed() != null) {
            throw new IllegalArgumentException("Amazon Bedrock models didn't support setting the seed");
        }

        var awsAccessKeyId = runContext.render(this.accessKeyId).as(String.class).orElseThrow(() -> new IllegalVariableEvaluationException("AWS Access Key ID cannot be null"));
        var awsSecretAccessKey = runContext.render(this.secretAccessKey).as(String.class).orElseThrow(() -> new IllegalVariableEvaluationException("AWS Secret Access Key cannot be null"));

        var credentials = AwsBasicCredentials.create(awsAccessKeyId, awsSecretAccessKey);
        var credentialsProvider = StaticCredentialsProvider.create(credentials);

        return BedrockChatModel.builder()
            .client(
                BedrockRuntimeClient.builder()
                    .credentialsProvider(credentialsProvider)
                    .build()
            )
            .modelId(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .defaultRequestParameters(BedrockChatRequestParameters.builder()
                .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
                .topK(runContext.render(configuration.getTopK()).as(Integer.class).orElse(null))
                .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
                .responseFormat(configuration.computeResponseFormat(runContext))
                .build()
            )
            .logRequests(runContext.render(configuration.getLogRequests()).as(Boolean.class).orElse(false))
            .logResponses(runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false))
            .logger(runContext.logger())
            .listeners(List.of(new TimingChatModelListener()))
            .build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        throw new UnsupportedOperationException("Amazon Bedrock didn't support image model");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        var modelType = runContext.render(this.modelType).as(AmazonBedrockEmbeddingModelType.class).orElseThrow();
        var modelName = runContext.render(this.getModelName()).as(String.class).orElseThrow();
        var bedrockRuntimeClient = BedrockRuntimeClient.builder()
            .credentialsProvider(
                StaticCredentialsProvider.create(
                    AwsBasicCredentials.create(
                        runContext.render(this.accessKeyId).as(String.class).orElseThrow(),
                        runContext.render(this.secretAccessKey).as(String.class).orElseThrow()
                    )
                )
            )
            .build();

        if (modelType == AmazonBedrockEmbeddingModelType.COHERE) {
            return BedrockCohereEmbeddingModel.builder()
                .client(bedrockRuntimeClient)
                .model(modelName)
                .build();
        } else if (modelType == AmazonBedrockEmbeddingModelType.TITAN) {
            return BedrockTitanEmbeddingModel.builder()
                .client(bedrockRuntimeClient)
                .model(modelName)
                .build();
        } else {
            throw new UnsupportedOperationException("Amazon Bedrock didn't support embedding model");
        }
    }

    enum AmazonBedrockEmbeddingModelType {
        COHERE,
        TITAN,
    }
}