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
    title = "Use Cohere models",
    description = "Connects to the Cohere OpenAI-compatible Compatibility API with API key and model name. Supports chat completion (including tool calling) and embeddings. Image generation is not available, and structured output follows Cohere's json_object format rather than OpenAI's strict JSON schema mode."
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with Cohere",
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
                          type: io.kestra.plugin.ai.provider.Cohere
                          apiKey: "{{ secret('COHERE_API_KEY') }}"
                          modelName: command-a-03-2025
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
public class Cohere extends OpenAICompliantProvider {
    private static final String BASE_URL = "https://api.cohere.ai/compatibility/v1";

    @Schema(
        title = "API base URL",
        description = "Base URL of the Cohere OpenAI-compatible API. Defaults to `https://api.cohere.ai/compatibility/v1`.",
        example = "https://api.cohere.ai/compatibility/v1"
    )
    @Builder.Default
    @PluginProperty(group = "connection")
    private Property<String> baseUrl = Property.ofValue(BASE_URL);
}
