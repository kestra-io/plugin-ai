package io.kestra.plugin.ai.provider;

import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.ai.langdock.LangdockModelFamily;
import io.kestra.plugin.ai.langdock.LangdockRegion;
import io.kestra.plugin.ai.langdock.LangdockRoutes;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import io.kestra.core.models.annotations.PluginProperty;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(
    title = "Use Langdock models",
    description = """
        Connects to Langdock's Completion API, which exposes OpenAI/Azure OpenAI-backed models on the \
        `OPENAI` route and Claude models on the `ANTHROPIC` route. Set `modelFamily` to match the \
        `modelName` you use: Claude models are only reachable when `modelFamily` is `ANTHROPIC`. Use the \
        `langdock.ListModels` task with the matching family to discover valid `modelName` values.

        Embeddings always go through the OpenAI route (only `text-embedding-ada-002` is supported there) \
        and require a workspace API key with the Embedding API scope — a personal key only works for chat \
        and model listing. Image generation is not offered by the Completion API.

        For a dedicated deployment, set `baseUrl` to the route root that matches the operation you are \
        using this provider for, e.g. `https://acme.langdock.com/api/public/openai/eu/v1` for chat/embeddings \
        on the OpenAI route, or `https://acme.langdock.com/api/public/anthropic/eu/v1/` for chat on the \
        Anthropic route; it then takes precedence over `region`."""
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with a Langdock-hosted OpenAI model",
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
                          type: io.kestra.plugin.ai.provider.Langdock
                          apiKey: "{{ secret('LANGDOCK_API_KEY') }}"
                          modelFamily: OPENAI
                          modelName: gpt-5.4-mini
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            }
        ),
        @Example(
            title = "AI agent using a Langdock-hosted Claude model with a Kestra tool",
            full = true,
            code = {
                """
                    id: agent_with_tool
                    namespace: company.ai

                    inputs:
                      - id: prompt
                        type: STRING
                        defaults: "Log the message 'Hello from Langdock!'"

                    tasks:
                      - id: agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        provider:
                          type: io.kestra.plugin.ai.provider.Langdock
                          apiKey: "{{ secret('LANGDOCK_API_KEY') }}"
                          modelFamily: ANTHROPIC
                          modelName: claude-sonnet-4-6-default
                        prompt: "{{ inputs.prompt }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.KestraTask
                            tasks:
                              - id: log
                                type: io.kestra.plugin.core.log.Log
                                message: "..."
                    """
            }
        ),
        @Example(
            title = "Ingest documents into a KV embedding store using Langdock embeddings",
            full = true,
            code = {
                """
                    id: document_ingestion
                    namespace: company.ai

                    tasks:
                      - id: ingest
                        type: io.kestra.plugin.ai.rag.IngestDocument
                        provider:
                          type: io.kestra.plugin.ai.provider.Langdock
                          apiKey: "{{ secret('LANGDOCK_WORKSPACE_API_KEY') }}"
                          modelName: text-embedding-ada-002
                        embeddings:
                          type: io.kestra.plugin.ai.embeddings.KestraKVStore
                        drop: true
                        fromExternalURLs:
                          - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/README.md
                    """
            }
        )
    }
)
public class Langdock extends ModelProvider {
    @Schema(title = "API Key")
    @NotNull
    @PluginProperty(secret = true, group = "main")
    private Property<String> apiKey;

    @Schema(
        title = "Model family",
        description = """
            Selects which Langdock Completion API route serves the request:
            - `OPENAI` (default): the OpenAI-compatible route, for OpenAI/Azure OpenAI-backed models.
            - `ANTHROPIC`: the Anthropic Messages-compatible route, required to reach Claude models.
            Ignored for embeddings, which always use the OpenAI route."""
    )
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<LangdockModelFamily> modelFamily = Property.ofValue(LangdockModelFamily.OPENAI);

    @Schema(
        title = "Region",
        description = "The Langdock region that serves the request. Ignored when a dedicated-deployment `baseUrl` is set."
    )
    @Builder.Default
    @PluginProperty(group = "connection")
    private Property<LangdockRegion> region = Property.ofValue(LangdockRegion.EU);

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        return chatModel(runContext, configuration, Duration.ofSeconds(120), Collections.emptyList());
    }

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout, List<ChatModelListener> additionalListeners)
        throws IllegalVariableEvaluationException {
        LangdockModelFamily rFamily = resolveModelFamily(runContext);
        String rBaseUrl = resolveChatBaseUrl(runContext, rFamily);

        return switch (rFamily) {
            case OPENAI -> openAiDelegate(rBaseUrl).chatModel(runContext, configuration, timeout, additionalListeners);
            case ANTHROPIC -> anthropicDelegate(runContext, rBaseUrl).chatModel(runContext, configuration, timeout, additionalListeners);
        };
    }

    @Override
    public ImageModel imageModel(RunContext runContext) {
        throw new UnsupportedOperationException("Langdock is currently not supported for image generation.");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        String rBaseUrl = resolveEmbeddingsBaseUrl(runContext);
        return openAiDelegate(rBaseUrl).embeddingModel(runContext);
    }

    private LangdockModelFamily resolveModelFamily(RunContext runContext) throws IllegalVariableEvaluationException {
        return runContext.render(modelFamily).as(LangdockModelFamily.class).orElse(LangdockModelFamily.OPENAI);
    }

    private LangdockRegion resolveRegion(RunContext runContext) throws IllegalVariableEvaluationException {
        return runContext.render(region).as(LangdockRegion.class).orElse(LangdockRegion.EU);
    }

    private String resolveChatBaseUrl(RunContext runContext, LangdockModelFamily family) throws IllegalVariableEvaluationException {
        String rBaseUrl = runContext.render(this.getBaseUrl()).as(String.class).orElse(null);
        if (rBaseUrl != null) {
            return LangdockRoutes.normalizeBaseUrl(rBaseUrl, family);
        }
        return LangdockRoutes.chatBaseUrl(resolveRegion(runContext), family);
    }

    // A `baseUrl` override always wins, whatever operation is being invoked: a task only ever uses one
    // of chatModel/embeddingModel/imageModel per provider instance, so there is no ambiguity about
    // which route a user-supplied dedicated-deployment URL is meant for.
    private String resolveEmbeddingsBaseUrl(RunContext runContext) throws IllegalVariableEvaluationException {
        String rBaseUrl = runContext.render(this.getBaseUrl()).as(String.class).orElse(null);
        if (rBaseUrl != null) {
            return LangdockRoutes.normalizeBaseUrl(rBaseUrl, LangdockModelFamily.OPENAI);
        }
        return LangdockRoutes.embeddingsBaseUrl(resolveRegion(runContext));
    }

    private LangdockOpenAiProvider openAiDelegate(String rBaseUrl) {
        return LangdockOpenAiProvider.builder()
            .type(LangdockOpenAiProvider.class.getName())
            .modelName(this.getModelName())
            .apiKey(this.apiKey)
            .baseUrl(Property.ofValue(rBaseUrl))
            .clientPem(this.getClientPem())
            .caPem(this.getCaPem())
            .build();
    }

    private LangdockAnthropicProvider anthropicDelegate(RunContext runContext, String rBaseUrl) throws IllegalVariableEvaluationException {
        String rApiKey = runContext.render(this.apiKey).as(String.class).orElseThrow();
        return LangdockAnthropicProvider.builder()
            .type(LangdockAnthropicProvider.class.getName())
            .modelName(this.getModelName())
            .apiKey(this.apiKey)
            .baseUrl(Property.ofValue(rBaseUrl))
            .clientPem(this.getClientPem())
            .caPem(this.getCaPem())
            .bearerToken(rApiKey)
            .build();
    }

    /**
     * Delegates to the existing OpenAI-compliant builder so Langdock's OpenAI route keeps every existing
     * validation and behaviour (topK rejection, strict JSON handling, returnThinking, PEM handling, ...).
     */
    @Getter
    @SuperBuilder
    private static final class LangdockOpenAiProvider extends OpenAICompliantProvider {
        // A class with no explicit constructor briefly looks like it has a no-arg one to Kestra's plugin
        // annotation processor, before Lombok's @SuperBuilder constructor replaces it — which is enough
        // for the processor to wrongly register this class as a standalone, selectable provider (visible
        // in build/classes/java/main/META-INF/services/io.kestra.core.models.Plugin). This unused,
        // never-called constructor pre-empts that by giving the class a real, non-empty-arg constructor
        // from the very first compiler pass.
        private LangdockOpenAiProvider(Void discriminator) {
        }
    }

    /**
     * Delegates to the existing Anthropic builder so Langdock's Anthropic route keeps every existing
     * validation and behaviour (seed/responseFormat rejection, thinking budget, prompt caching, ...),
     * adding only the Bearer header Langdock requires on top of Anthropic's own {@code x-api-key} header.
     */
    @Getter
    @SuperBuilder
    private static final class LangdockAnthropicProvider extends Anthropic {
        private String bearerToken;

        // See the identical constructor on LangdockOpenAiProvider for why this is needed.
        private LangdockAnthropicProvider(Void discriminator) {
        }

        @Override
        protected Map<String, String> customHeaders(RunContext runContext) {
            return Map.of("Authorization", "Bearer " + bearerToken);
        }
    }
}
