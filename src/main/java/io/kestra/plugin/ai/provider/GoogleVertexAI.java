package io.kestra.plugin.ai.provider;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.gcp.shared.CredentialService;
import io.kestra.plugin.gcp.shared.GcpInterface;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiEmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.vertexai.VertexAiImageModel;
import dev.langchain4j.model.vertexai.gemini.SchemaHelper;
import dev.langchain4j.model.vertexai.gemini.VertexAiGeminiChatModel;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Use Google Vertex AI models",
    description = "Calls Vertex AI Gemini chat, embeddings, or images using project, location, and endpoint settings. Authenticates like the GCP plugin: `serviceAccount` when set, otherwise Application Default Credentials, optionally impersonating `impersonatedServiceAccount`; ensure response formats are supported by the selected model/region."
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with Google Vertex AI",
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
                          type: io.kestra.plugin.ai.provider.GoogleVertexAI
                          modelName: gemini-3.5-flash-lite
                          location: your-google-cloud-region
                          projectId: your-google-cloud-project-id
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            }
        ),
        @Example(
            title = "Chat completion with Google Vertex AI using a service account",
            full = true,
            code = {
                """
                    id: chat_completion_service_account
                    namespace: company.ai

                    tasks:
                      - id: chat_completion
                        type: io.kestra.plugin.ai.completion.ChatCompletion
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleVertexAI
                          modelName: gemini-3.5-flash-lite
                          location: your-google-cloud-region
                          projectId: your-google-cloud-project-id
                          serviceAccount: "{{ secret('GCP_SERVICE_ACCOUNT_JSON') }}"
                        messages:
                          - type: USER
                            content: What is the capital of France?
                    """
            }
        )
    },
    aliases = "io.kestra.plugin.langchain4j.provider.GoogleVertexAI"
)
public class GoogleVertexAI extends ModelProvider implements GcpInterface {
    private static final String GLOBAL_LOCATION = "global";
    // Chat's VertexAI SDK appends ":443" itself; the image SDK expects host:port.
    private static final String GLOBAL_HOST = "aiplatform.googleapis.com";
    // `text-embedding-*` models reject requests over 20,000 tokens, and each text is truncated to 2,048 tokens: 9 texts always fit.
    // `gemini-embedding-*` models have no such limit and keep the SDK default of 100 texts per request.
    static final int TEXT_EMBEDDING_MAX_SEGMENTS_PER_BATCH = 9;

    @Schema(
        title = "Endpoint URL",
        description = "Vertex AI API endpoint for image and embedding models, as `host:port` or a full `https://` URL. Not set by default, in which case the endpoint is derived from `location` (`aiplatform.googleapis.com` for `global`). Must not be set for chat models, which always use Gemini.",
        example = "us-central1-aiplatform.googleapis.com:443"
    )
    @PluginProperty(group = "main")
    private Property<String> endpoint;

    @Schema(
        title = "Project location",
        description = "Google Cloud region hosting the Vertex AI model, or `global` to use the global endpoint `aiplatform.googleapis.com`. No default: this property is required for chat models.",
        example = "us-central1"
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<String> location;

    @Schema(
        title = "Project ID",
        description = "Google Cloud project ID that owns the Vertex AI resources. When not set, it is taken from the service account credentials.",
        example = "my-gcp-project"
    )
    @PluginProperty(group = "main")
    private Property<String> projectId;

    @Schema(
        title = "Project ID (deprecated)",
        description = "Deprecated, use `projectId` instead. Only used when `projectId` is not set.",
        example = "my-gcp-project",
        deprecated = true
    )
    @PluginProperty(group = "main")
    @Deprecated
    private Property<String> project;

    @Schema(
        title = "Service account JSON key",
        description = "Content of a Google Cloud service account JSON key. When not set, Application Default Credentials are used. Not supported for image models, which always use Application Default Credentials."
    )
    @PluginProperty(secret = true, group = "connection")
    private Property<String> serviceAccount;

    @Schema(
        title = "Service account to impersonate",
        description = "Email of the service account to impersonate. Not supported for image models."
    )
    @PluginProperty(secret = true, group = "connection")
    private Property<String> impersonatedServiceAccount;

    @Schema(
        title = "OAuth scopes",
        description = "OAuth scopes requested for the credentials. Defaults to `https://www.googleapis.com/auth/cloud-platform`. Not supported for image models, which always use Application Default Credentials with their default scopes."
    )
    @PluginProperty(group = "connection")
    @Builder.Default
    private Property<List<String>> scopes = Property.ofValue(Collections.singletonList("https://www.googleapis.com/auth/cloud-platform"));

    @Override
    public Property<String> getProjectId() {
        return this.projectId != null ? this.projectId : this.project;
    }

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        return chatModel(runContext, configuration, Duration.ofSeconds(120), Collections.emptyList());
    }

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout, List<ChatModelListener> additionalListeners)
        throws IllegalVariableEvaluationException {
        if (this.endpoint != null) {
            throw new IllegalArgumentException("The `endpoint` property cannot be used for the Chat Model which uses Gemini only.");
        }

        var allListeners = new ArrayList<ChatModelListener>();
        allListeners.add(new TimingChatModelListener());
        allListeners.addAll(additionalListeners);

        var responseFormat = configuration.computeResponseFormat(runContext);

        var connection = connection(runContext);
        var renderedLocation = runContext.render(this.location).as(String.class).orElseThrow();

        return VertexAiGeminiChatModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .location(renderedLocation)
            .apiEndpoint(isGlobal(renderedLocation) ? GLOBAL_HOST : null)
            .project(projectId(runContext, connection))
            .credentials(connection.credentials())
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).map(d -> d.floatValue()).orElse(null))
            .topK(runContext.render(configuration.getTopK()).as(Integer.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).map(d -> d.floatValue()).orElse(null))
            .seed(runContext.render(configuration.getSeed()).as(Integer.class).orElse(null))
            .logRequests(runContext.render(configuration.getLogRequests()).as(Boolean.class).orElse(false))
            .logResponses(runContext.render(configuration.getLogResponses()).as(Boolean.class).orElse(false))
            .responseMimeType(responseFormat.type() == ResponseFormatType.JSON ? "application/json" : null)
            .responseSchema(responseFormat.jsonSchema() != null ? SchemaHelper.from(responseFormat.jsonSchema().rootElement()) : null)
            .listeners(allListeners)
            .maxOutputTokens(runContext.render(configuration.getMaxToken()).as(Integer.class).orElse(null))
            .build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        if (this.serviceAccount != null || this.impersonatedServiceAccount != null) {
            throw new IllegalArgumentException(
                "The `serviceAccount` and `impersonatedServiceAccount` properties cannot be used for the Image Model, use Application Default Credentials instead."
            );
        }

        var renderedLocation = runContext.render(this.location).as(String.class).orElse(null);

        return VertexAiImageModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .endpoint(endpoint(runContext, renderedLocation))
            .location(renderedLocation)
            .project(projectId(runContext, connection(runContext)))
            .build();
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        var connection = connection(runContext);
        var modelName = runContext.render(this.getModelName()).as(String.class).orElseThrow();

        // google-genai embeds through `predict` only: langchain4j-vertex-ai calls `countTokens` first, which isn't served on `global`.
        // It also resolves the `global` location to `aiplatform.googleapis.com` itself.
        return GoogleGenAiEmbeddingModel.builder()
            .modelName(modelName)
            .location(runContext.render(this.location).as(String.class).orElse(null))
            .projectId(projectId(runContext, connection))
            .googleCredentials(connection.credentials())
            .apiEndpoint(apiEndpoint(runContext.render(this.endpoint).as(String.class).orElse(null)))
            .maxSegmentsPerBatch(modelName.startsWith("gemini-embedding") ? null : TEXT_EMBEDDING_MAX_SEGMENTS_PER_BATCH)
            .build();
    }

    CredentialService.GcpConnection connection(RunContext runContext) throws IllegalVariableEvaluationException {
        try {
            return CredentialService.connection(runContext, this);
        } catch (IOException e) {
            throw new UncheckedIOException("Unable to load the Google Cloud credentials", e);
        }
    }

    // The SDKs derive `<location>-aiplatform.googleapis.com`, which doesn't exist for the `global` location.
    static boolean isGlobal(String location) {
        return GLOBAL_LOCATION.equalsIgnoreCase(location);
    }

    String endpoint(RunContext runContext, String renderedLocation) throws IllegalVariableEvaluationException {
        return runContext.render(this.endpoint).as(String.class)
            .orElse(isGlobal(renderedLocation) ? GLOBAL_HOST + ":443" : null);
    }

    // The `endpoint` property is documented as host:port, while google-genai expects a base URL.
    static String apiEndpoint(String endpoint) {
        if (endpoint == null || endpoint.contains("://")) {
            return endpoint;
        }
        return "https://" + (endpoint.endsWith(":443") ? endpoint.substring(0, endpoint.length() - ":443".length()) : endpoint);
    }

    private static String projectId(RunContext runContext, CredentialService.GcpConnection connection) throws IllegalVariableEvaluationException {
        return runContext.render(connection.projectId()).as(String.class)
            .orElseThrow(() -> new IllegalArgumentException("The `projectId` property is required when it cannot be taken from the service account credentials."));
    }
}
