package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.oracle.bmc.Region;
import com.oracle.bmc.auth.AuthenticationDetailsProvider;
import com.oracle.bmc.auth.ConfigFileAuthenticationDetailsProvider;
import dev.langchain4j.community.model.oracle.oci.genai.OciGenAiChatModel;
import dev.langchain4j.community.model.oracle.oci.genai.OciGenAiCohereChatModel;
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
import org.apache.commons.lang3.Strings;

import java.io.IOException;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(
    title = "OciGenAI Model Provider"
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with OciGenAI",
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
                          type: io.kestra.plugin.ai.provider.OciGenAI
                          region: "{{ kv('OCI_GENAI_MODEL_REGION_PROPERTY') }}"
                          compartmentId: "{{ kv('OCI_GENAI_COMPARTMENT_ID_PROPERTY') }}"
                          authProvider: "{{ kv('OCI_GENAI_CONFIG_PROFILE_PROPERTY') }}"
                          modelName: oracle.chat.gpt-3.5
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                          - type: USER
                            content: "{{inputs.prompt}}"
                    """
            }
        )
    }
)
public class OciGenAI extends ModelProvider {

    private static final String DEFAULT = "DEFAULT";
    @Schema(title = "OCID of OCI Compartment with the model")
    @NotNull
    private Property<String> compartmentId;

    @Schema(title = "OCI Region to connect the client to")
    @NotNull
    private Property<String> region;

    @Schema(title = "OCI SDK Authentication provider")
    private Property<String> authProvider;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        var rModelName = runContext.render(this.getModelName()).as(String.class).orElseThrow();
        if (Strings.CI.contains(rModelName, "cohere")) {
            return buildCohereChatModel(runContext, configuration, rModelName);
        }
        return buildGenAiChatModel(runContext, configuration, rModelName);
    }

    @Override
    public ImageModel imageModel(RunContext runContext) {
        throw new UnsupportedOperationException("OciGenAI is currently not supported for image generation.");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        throw new UnsupportedOperationException("OciGenAI is currently not supported for image generation.");
    }

    private AuthenticationDetailsProvider createAuthProvider(final RunContext runContext) throws IllegalVariableEvaluationException {
        var configProfile = runContext.render(this.getAuthProvider()).as(String.class).orElse(DEFAULT);
        try {
            return new ConfigFileAuthenticationDetailsProvider(configProfile);
        } catch (final IOException e) {
            throw new RuntimeException("Error when setting up auth provider.", e);
        }
    }

    private ChatModel buildGenAiChatModel(RunContext runContext, ChatConfiguration configuration, String modelName) throws IllegalVariableEvaluationException {
        return OciGenAiChatModel.builder()
            .modelName(modelName)
            .compartmentId(runContext.render(this.getCompartmentId()).as(String.class).orElseThrow())
            .region(Region.fromRegionCodeOrId(runContext.render(this.getRegion()).as(String.class).orElseThrow()))
            .authProvider(createAuthProvider(runContext))
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .topK(runContext.render(configuration.getTopK()).as(Integer.class).orElseThrow())
            .seed(runContext.render(configuration.getSeed()).as(Integer.class).orElse(null))
            .maxTokens(runContext.render(configuration.getMaxToken()).as(Integer.class).orElse(null))
            .build();
    }

    private ChatModel buildCohereChatModel(RunContext runContext, ChatConfiguration configuration, String modelName) throws IllegalVariableEvaluationException {
        return OciGenAiCohereChatModel.builder()
            .modelName(modelName)
            .compartmentId(runContext.render(this.getCompartmentId()).as(String.class).orElseThrow())
            .region(Region.fromRegionCodeOrId(runContext.render(this.getRegion()).as(String.class).orElseThrow()))
            .authProvider(createAuthProvider(runContext))
            .temperature(runContext.render(configuration.getTemperature()).as(Double.class).orElse(null))
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .topK(runContext.render(configuration.getTopK()).as(Integer.class).orElseThrow())
            .seed(runContext.render(configuration.getSeed()).as(Integer.class).orElse(null))
            .maxTokens(runContext.render(configuration.getMaxToken()).as(Integer.class).orElse(null))
            .build();
    }
}
