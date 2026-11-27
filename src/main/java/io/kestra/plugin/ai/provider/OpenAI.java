package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;

import dev.langchain4j.model.openai.internal.OpenAiUtils;
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
    title = "Use OpenAI models",
    description = "Connects to OpenAI-compatible endpoints (defaults to api.openai.com) for chat, embeddings, or images. Requires API key; override `baseUrl` for Azure-compatible or proxy setups."
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
                        type: io.kestra.plugin.ai.completion.ChatCompletion
                        provider:
                          type: io.kestra.plugin.ai.provider.OpenAI
                          apiKey: "{{ secret('OPENAI_API_KEY') }}"
                          modelName: gpt-5-mini
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            }
        ),
        @Example(
            title = "Chat completion with an OpenAI-compatible endpoint (Groq)",
            full = true,
            code = {
                """
                    id: chat_completion_openai_compatible
                    namespace: company.ai

                    inputs:
                      - id: prompt
                        type: STRING

                    tasks:
                      - id: chat_completion
                        type: io.kestra.plugin.ai.completion.ChatCompletion
                        provider:
                          type: io.kestra.plugin.ai.provider.OpenAI
                          apiKey: "{{ secret('GROQ_API_KEY') }}"
                          baseUrl: https://api.groq.com/openai/v1
                          modelName: llama-3.3-70b-versatile
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely.
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            }
        )
    },
    aliases = "io.kestra.plugin.langchain4j.provider.OpenAI"
)
public class OpenAI extends OpenAICompliantProvider {
    @Schema(
        title = "API base URL",
        description = "Base URL of the OpenAI-compatible API. Override it to target Azure OpenAI, a proxy, a self-hosted gateway, or another vendor that exposes an OpenAI-compatible API, for example Groq. Support for tool calling, structured output, and image generation depends on the vendor. Defaults to `https://api.openai.com/v1`.",
        example = "https://api.openai.com/v1"
    )
    @Builder.Default
    @PluginProperty(group = "connection")
    private Property<String> baseUrl = Property.ofValue(OpenAiUtils.DEFAULT_OPENAI_URL);
}
