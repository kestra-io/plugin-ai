package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import io.kestra.core.models.annotations.PluginProperty;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Use Cheaper Inference models",
    description = "Connects to the Cheaper Inference OpenAI-compatible API with API key and model name. Cheaper Inference is an LLM gateway: one API key gives access to models from several labs. Supports chat completion (including tool calling) and embeddings. Image generation is not available."
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with Cheaper Inference",
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
                          type: io.kestra.plugin.ai.provider.CheaperInference
                          apiKey: "{{ secret('CHEAPER_INFERENCE_API_KEY') }}"
                          modelName: gpt-5.4-mini
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            }
        )
    }
)
public class CheaperInference extends OpenAICompliantProvider {
    private static final String BASE_URL = "https://api.cheaperinference.com/v1";

    @Schema(
        title = "API base URL",
        description = "Base URL of the Cheaper Inference OpenAI-compatible API. Defaults to `https://api.cheaperinference.com/v1`.",
        example = "https://api.cheaperinference.com/v1"
    )
    @Builder.Default
    @PluginProperty(group = "connection")
    private Property<String> baseUrl = Property.ofValue(BASE_URL);
}
