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

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "HuggingFace Model Provider"
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with HuggingFace",
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
                      type: io.kestra.plugin.ai.provider.HuggingFace
                      apiKey: "{{ kv('HUGGING_FACE_API_KEY') }}"
                      modelName: HuggingFaceTB/SmolLM3-3B:hf-inference
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
public class HuggingFace extends OpenAICompliantProvider {
    private static final String BASE_URL = "https://router.huggingface.co/v1";

    @Schema(title = "API base URL")
    @Builder.Default
    private Property<String> baseUrl = Property.ofValue(BASE_URL);
}
