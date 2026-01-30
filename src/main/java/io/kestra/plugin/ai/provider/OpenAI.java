package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.model.openai.internal.OpenAiUtils;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
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
                      apiKey: "{{ kv('OPENAI_API_KEY') }}"
                      modelName: gpt-5-mini
                    messages:
                      - type: SYSTEM
                        content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                      - type: USER
                        content: "{{inputs.prompt}}"
                """
            }
        )
    },
    aliases = "io.kestra.plugin.langchain4j.provider.OpenAI"
)
public class OpenAI extends OpenAICompliantProvider {
    @Schema(title = "API base URL")
    @Builder.Default
    private Property<String> baseUrl = Property.ofValue(OpenAiUtils.DEFAULT_OPENAI_URL);
}
