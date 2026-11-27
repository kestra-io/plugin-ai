package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Use xAI Grok models",
    description = "Connects to xAI's OpenAI-compatible API using an API key and model name for chat tasks."
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with xAI Grok",
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
                          type: io.kestra.plugin.ai.provider.XAI
                          apiKey: "{{ secret('XAI_API_KEY') }}"
                          modelName: grok-3
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant. Answer concisely.
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            }
        )
    },
    aliases = "io.kestra.plugin.langchain4j.provider.XAI"
)
public class XAI extends OpenAICompliantProvider {
    private static final String BASE_URL = "https://api.x.ai/v1";

    @Schema(
        title = "API base URL",
        description = "Base URL of the xAI OpenAI-compatible API. Defaults to `https://api.x.ai/v1`.",
        example = "https://api.x.ai/v1"
    )
    @Builder.Default
    @PluginProperty(group = "connection")
    private Property<String> baseUrl = Property.ofValue(BASE_URL);
}