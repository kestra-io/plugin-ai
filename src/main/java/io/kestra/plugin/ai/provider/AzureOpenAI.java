package io.kestra.plugin.ai.provider;

import com.azure.core.credential.TokenCredential;
import com.azure.identity.ClientSecretCredentialBuilder;
import com.azure.identity.DefaultAzureCredentialBuilder;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.model.azure.AzureOpenAiChatModel;
import dev.langchain4j.model.azure.AzureOpenAiEmbeddingModel;
import dev.langchain4j.model.azure.AzureOpenAiImageModel;
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
import org.apache.commons.lang3.StringUtils;

import java.util.List;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(
        title = "Use Azure OpenAI deployments",
        description = "Targets Azure-hosted OpenAI models via the resource endpoint and deployment name. Supports API key or AAD client credentials; set `apiVersion` when required by the deployment."
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with Azure OpenAI",
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
                      type: io.kestra.plugin.ai.provider.AzureOpenAI
                      apiKey: "{{ kv('AZURE_API_KEY') }}"
                      endpoint: https://your-resource.openai.azure.com/
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
    aliases = "io.kestra.plugin.langchain4j.provider.AzureOpenAI"
)
public class AzureOpenAI extends ModelProvider {

    @Schema(title = "API Key")
    private Property<String> apiKey;

    @Schema(title = "API endpoint", description = "The Azure OpenAI endpoint in the format: https://{resource}.openai.azure.com/")
    @NotNull
    private Property<String> endpoint;

    @Schema(title = "API version")
    private Property<String> serviceVersion;

    @Schema(title = "Tenant ID")
    private Property<String> tenantId;

    @Schema(title = "Client ID")
    private Property<String> clientId;

    @Schema(title = "Client secret")
    private Property<String> clientSecret;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        if (configuration.getTopK() != null) {
            throw new IllegalArgumentException("Azure OpenAI models do not support setting the topK");
        }

        var seed = runContext.render(configuration.getSeed()).as(Integer.class).orElse(null);

        var apiKey = runContext.render(this.apiKey).as(String.class).orElse(null);
        var tenantId = runContext.render(this.tenantId).as(String.class).orElse(null);
        var clientId = runContext.render(this.clientId).as(String.class).orElse(null);
        var clientSecret = runContext.render(this.clientSecret).as(String.class).orElse(null);
        var logRequestAndResponses = runContext.render(configuration.getLogRequests()).as(Boolean.class).orElse(false) ||
            runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false);

        AzureOpenAiChatModel.Builder defaultBuilder = AzureOpenAiChatModel.builder()
            .deploymentName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .endpoint(runContext.render(this.getEndpoint()).as(String.class).orElseThrow())
            .serviceVersion(runContext.render(this.getServiceVersion()).as(String.class).orElse(null))
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .seed(seed != null ? seed.longValue() : null)
            .logRequestsAndResponses(logRequestAndResponses)
            .responseFormat(configuration.computeResponseFormat(runContext))
            .listeners(List.of(new TimingChatModelListener()))
            .maxCompletionTokens(runContext.render(configuration.getMaxToken()).as(Integer.class).orElse(null));

        if (apiKey != null) {
            return defaultBuilder
                .apiKey(apiKey)
                .build();
        } else if (tenantId != null && clientId != null && clientSecret != null) {
            return defaultBuilder
                .tokenCredential(credentials(runContext, tenantId, clientId, clientSecret))
                .build();
        } else {
            throw new IllegalArgumentException("You need to set an API Key or a tenantId, clientId and clientSecret");
        }
    }

    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        var apiKey = runContext.render(this.apiKey).as(String.class).orElse(null);
        var tenantId = runContext.render(this.tenantId).as(String.class).orElse(null);
        var clientId = runContext.render(this.clientId).as(String.class).orElse(null);
        var clientSecret = runContext.render(this.clientSecret).as(String.class).orElse(null);

        if (apiKey != null) {
            return AzureOpenAiImageModel.builder()
                    .apiKey(apiKey)
                    .deploymentName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
                    .endpoint(runContext.render(this.getEndpoint()).as(String.class).orElseThrow())
                    .serviceVersion(runContext.render(this.getServiceVersion()).as(String.class).orElse(null))
                    .build();
        } else if (tenantId != null && clientId != null && clientSecret != null) {
            return AzureOpenAiImageModel.builder()
                    .tokenCredential(credentials(runContext, tenantId, clientId, clientSecret))
                    .deploymentName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
                    .endpoint(runContext.render(this.getEndpoint()).as(String.class).orElseThrow())
                    .serviceVersion(runContext.render(this.getServiceVersion()).as(String.class).orElse(null))
                    .build();
        } else {
            throw new IllegalArgumentException("You need to set an API Key or a tenantId, clientId and clientSecret");
        }
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        var apiKey = runContext.render(this.apiKey).as(String.class).orElse(null);
        var tenantId = runContext.render(this.tenantId).as(String.class).orElse(null);
        var clientId = runContext.render(this.clientId).as(String.class).orElse(null);
        var clientSecret = runContext.render(this.clientSecret).as(String.class).orElse(null);

        if (apiKey != null) {
            return AzureOpenAiEmbeddingModel.builder()
                    .apiKey(apiKey)
                    .deploymentName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
                    .endpoint(runContext.render(this.getEndpoint()).as(String.class).orElseThrow())
                    .serviceVersion(runContext.render(this.getServiceVersion()).as(String.class).orElse(null))
                    .build();
        } else {
            return AzureOpenAiEmbeddingModel.builder()
                    .tokenCredential(credentials(runContext, tenantId, clientId, clientSecret))
                    .deploymentName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
                    .endpoint(runContext.render(this.getEndpoint()).as(String.class).orElseThrow())
                    .serviceVersion(runContext.render(this.getServiceVersion()).as(String.class).orElse(null))
                    .build();
        }
    }

    private TokenCredential credentials(RunContext runContext, String tenantId, String clientId, String clientSecret) {

        if (StringUtils.isNotBlank(tenantId) && StringUtils.isNotBlank(clientId) && StringUtils.isNotBlank(clientSecret)) {
            runContext.logger().info("Authentication is using Client Secret Credentials");
            return new ClientSecretCredentialBuilder()
                    .clientId(clientId)
                    .tenantId(tenantId)
                    .clientSecret(clientSecret)
                    .build();
        }

        runContext.logger().info("Authentication is using Default Azure Credentials");
        return new DefaultAzureCredentialBuilder().tenantId(tenantId).build();
    }
}
