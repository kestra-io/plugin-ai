package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.workersai.WorkersAiChatModel;
import dev.langchain4j.model.workersai.WorkersAiEmbeddingModel;
import dev.langchain4j.model.workersai.WorkersAiImageModel;
import dev.langchain4j.model.workersai.WorkersAiImageModelName;
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

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(
    title = "Use Cloudflare Workers AI models",
    description = "Invokes Workers AI chat, embedding, and image models using account ID and API key. Ensure the selected model is available in your account/region."
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with WorkersAI",
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
                      type: io.kestra.plugin.ai.provider.WorkersAI
                      accountId: "{{ kv('WORKERS_AI_ACCOUNT_ID') }}"
                      apiKey: "{{ kv('WORKERS_AI_API_KEY') }}"
                      modelName: @cf/meta/llama-2-7b-chat-fp16
                    messages:
                      - type: SYSTEM
                        content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                      - type: USER
                        content: "{{inputs.prompt}}"
                """
            }
        )
    },
    aliases = "io.kestra.plugin.langchain4j.provider.WorkersAI"
)
public class WorkersAI extends ModelProvider {
    @Schema(title = "API Key")
    @NotNull
    private Property<String> apiKey;

    @Schema(
        title = "Account Identifier",
        description = "Unique identifier assigned to an account"
    )
    @NotNull
    private Property<String> accountId;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        return WorkersAiChatModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .accountId(runContext.render(this.accountId).as(String.class).orElseThrow())
            .apiToken(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return WorkersAiImageModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .accountId(runContext.render(this.accountId).as(String.class).orElseThrow())
            .apiToken(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .build();
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return WorkersAiEmbeddingModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .accountId(runContext.render(this.accountId).as(String.class).orElseThrow())
            .apiToken(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .build();
    }

}
