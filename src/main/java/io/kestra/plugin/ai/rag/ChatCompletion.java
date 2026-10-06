package io.kestra.plugin.ai.rag;

import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Metric;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.ListUtils;
import io.kestra.plugin.ai.AIUtils;
import io.kestra.plugin.ai.TokenBudgetChatModel;
import io.kestra.plugin.ai.domain.*;
import io.kestra.plugin.ai.guardrail.GuardrailsEvaluator;
import io.kestra.plugin.ai.provider.TimingChatModelListener;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.guardrail.GuardrailException;
import dev.langchain4j.guardrail.InputGuardrailException;
import dev.langchain4j.guardrail.OutputGuardrailException;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.annotation.Nullable;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;

import static io.kestra.core.utils.Rethrow.throwFunction;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Run RAG chat with retrievers/tools",
    description = """
        Combines chat, an embedding-store retriever, optional content retrievers, and optional tools. Requires chat and embedding providers. Retrievers always supply context; tools run only when invoked by the model. Retriever limits (`maxResults`, `minScore`) filter retrieved chunks."""
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = """
                Chat with your data using Retrieval Augmented Generation (RAG). This flow will index documents and use the RAG Chat task to interact with your data using natural language prompts. The flow contrasts prompts to LLM with and without RAG. The Chat with RAG retrieves embeddings stored in the KV Store and provides a response grounded in data rather than hallucinating.
                WARNING: the Kestra KV embedding store is for quick prototyping only, as it stores the embedding vectors in Kestra's KV store and loads them all into memory.
                """,
            code = """
                id: rag
                namespace: company.ai

                tasks:
                  - id: ingest
                    type: io.kestra.plugin.ai.rag.IngestDocument
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-001
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    drop: true
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md

                  - id: parallel
                    type: io.kestra.plugin.core.flow.Parallel
                    tasks:
                      - id: chat_without_rag
                        type: io.kestra.plugin.ai.completion.ChatCompletion
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-3.5-flash-lite
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        messages:
                          - type: USER
                            content: Which features were released in Kestra 0.24?

                      - id: chat_with_rag
                        type: io.kestra.plugin.ai.rag.ChatCompletion
                        chatProvider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-3.5-flash-lite
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        embeddingProvider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-embedding-001
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        embeddings:
                          type: io.kestra.plugin.ai.embeddings.KestraKVStore
                        systemMessage: You are a helpful assistant that can answer questions about Kestra.
                        prompt: Which features were released in Kestra 0.24?"""
        ),
        @Example(
            full = true,
            title = "RAG chat with a web search content retriever (answers grounded in search results)",
            code = """
                id: rag_with_websearch_content_retriever
                namespace: company.ai

                tasks:
                  - id: chat_with_rag_and_websearch_content_retriever
                    type: io.kestra.plugin.ai.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-3.5-flash-lite
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.TavilyWebSearch
                        apiKey: "{{ secret('TAVILY_API_KEY') }}"
                    systemMessage: You are a helpful assistant that can answer questions about Kestra.
                    prompt: What is the latest release of Kestra?"""
        ),
        @Example(
            full = true,
            title = "Store chat memory as a Kestra KV pair",
            code = """
                id: chat_with_memory
                namespace: company.ai

                inputs:
                  - id: first
                    type: STRING
                    defaults: Hello, my name is John and I'm from Paris

                  - id: second
                    type: STRING
                    defaults: What's my name and where do I live?

                tasks:
                  - id: first
                    type: io.kestra.plugin.ai.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-3.5-flash-lite
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddingProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-001
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    memory:
                      type: io.kestra.plugin.ai.memory.KestraKVStore
                      ttl: PT1M
                    systemMessage: You are a helpful assistant, answer concisely
                    prompt: "{{ inputs.first }}"

                  - id: second
                    type: io.kestra.plugin.ai.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-3.5-flash-lite
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddingProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-001
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    memory:
                      type: io.kestra.plugin.ai.memory.KestraKVStore
                    systemMessage: You are a helpful assistant, answer concisely
                    prompt: "{{ inputs.second }}"
                """
        ),
        @Example(
            full = true,
            title = """
                Classify recent Kestra releases into MINOR or PATCH using a JSON schema.
                Note: not all LLMs support structured outputs, or they may not support them when combined with tools like web search.
                This example uses Mistral, which supports structured output with content retrievers.""",
            code = """
                id: chat_with_structured_output
                namespace: company.ai

                tasks:
                  - id: categorize_releases
                    type: io.kestra.plugin.ai.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.ai.provider.MistralAI
                      apiKey: "{{ secret('MISTRAL_API_KEY') }}"
                      modelName: open-mistral-7b

                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.TavilyWebSearch
                        apiKey: "{{ secret('TAVILY_API_KEY') }}"
                        maxResults: 8

                    chatConfiguration:
                      responseFormat:
                        type: JSON
                        jsonSchema:
                          type: object
                          required: ["releases"]
                          properties:
                            releases:
                              type: array
                              minItems: 1
                              items:
                                type: object
                                additionalProperties: false
                                required: ["version", "date", "semver"]
                                properties:
                                  version:
                                    type: string
                                    description: "Release tag, e.g., 0.24.0"
                                  date:
                                    type: string
                                    description: "Release date"
                                  semver:
                                    type: string
                                    enum: ["MINOR", "PATCH"]
                                  summary:
                                    type: string
                                    description: "Short plain-text summary (optional)"

                    systemMessage: |
                      You are a release analyst. Use the Tavily web retriever to find recent Kestra releases.
                      Determine each release's SemVer category:
                        - MINOR: new features, no major breaking changes (y in x.Y.z)
                        - PATCH: bug fixes/patches only (z in x.y.Z)
                      Return ONLY valid JSON matching the schema. No prose, no extra keys.

                    prompt: |
                      Find most recent Kestra releases (within the last ~6 months).
                      Output their version, release date, semver category, and a one-line summary."""
        )
    },
    metrics = {
        @Metric(
            name = "input.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) input token count"
        ),
        @Metric(
            name = "output.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) output token count"
        ),
        @Metric(
            name = "total.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) total token count"
        ),
        @Metric(
            name = "ai.agent.tool.calls",
            type = Counter.TYPE,
            unit = "calls",
            description = "Number of AI tool invocations during agent execution, tagged by tool class name"
        ),
        @Metric(
            name = "ai.provider.calls",
            type = Counter.TYPE,
            unit = "calls",
            description = "Number of times a chat or embedding model is obtained from a provider, tagged by provider class name"
        ),
        @Metric(
            name = "ai.embedding.store.calls",
            type = Counter.TYPE,
            unit = "calls",
            description = "Number of times an embedding store is used, tagged by store class name"
        )
    },
    aliases = "io.kestra.plugin.langchain4j.rag.ChatCompletion"
)
public class ChatCompletion extends Task implements RunnableTask<ChatCompletion.Output> {

    @Schema(
        title = "System message",
        description = "Instruction setting the assistant's role, tone and constraints for this task. Not set by default.",
        example = "You are a support assistant. Answer only from the retrieved documents, and say so when they do not cover the question."
    )
    @PluginProperty(group = "main")
    protected Property<String> systemMessage;

    @Schema(
        title = "User prompt",
        description = "Question asked of the model, which is also the query used to retrieve relevant documents. No default: this property is required.",
        example = "{{ inputs.question }}"
    )
    @NotNull
    @PluginProperty(group = "main")
    protected Property<String> prompt;

    @Schema(
        title = "Embedding store",
        description = "Vector store searched for documents relevant to the prompt. Optional when at least one entry is provided in `contentRetrievers`, required otherwise.",
        example = "{type: \"io.kestra.plugin.ai.embeddings.KestraKVStore\"}"
    )
    @PluginProperty(group = "advanced")
    private EmbeddingStoreProvider embeddings;

    @Schema(
        title = "Embedding model provider",
        description = "Model provider used to embed the prompt before searching the store. Defaults to `chatProvider`, which must then support embeddings. It should use the same embedding model that was used at ingestion time.",
        example = "{type: \"io.kestra.plugin.ai.provider.GoogleGemini\", apiKey: \"{{ secret('GEMINI_API_KEY') }}\", modelName: \"gemini-embedding-001\"}"
    )
    @PluginProperty(group = "advanced")
    private ModelProvider embeddingProvider;

    @Schema(
        title = "Chat model provider",
        description = "Model provider that generates the answer from the retrieved context. No default: this property is required.",
        example = "{type: \"io.kestra.plugin.ai.provider.GoogleGemini\", apiKey: \"{{ secret('GEMINI_API_KEY') }}\", modelName: \"gemini-3.5-flash-lite\"}"
    )
    @NotNull
    @PluginProperty(group = "main")
    private ModelProvider chatProvider;

    @Schema(
        title = "Chat configuration",
        description = "Chat model settings (temperature, response format, token limits, and so on). Defaults to an empty configuration, so the provider's own defaults apply.",
        example = "{temperature: 0.3, maxToken: 1024}"
    )
    @NotNull
    @PluginProperty(group = "advanced")
    @Builder.Default
    private ChatConfiguration chatConfiguration = ChatConfiguration.empty();

    @Schema(
        title = "Content retriever configuration",
        description = "How many documents the embedding store returns and how similar they must be, through `maxResults` and `minScore`. Defaults to `maxResults: 3` and `minScore: 0.0`.",
        example = "{maxResults: 5, minScore: 0.7}"
    )
    @NotNull
    @PluginProperty(group = "advanced")
    @Builder.Default
    private ContentRetrieverConfiguration contentRetrieverConfiguration = ContentRetrieverConfiguration.builder().build();

    @Schema(
        title = "Additional content retrievers",
        description = "Retrievers queried alongside the embedding store, whose results are always injected into the context, unlike tools, which the LLM calls only when it decides to. Not set by default.",
        example = "[{type: \"io.kestra.plugin.ai.retriever.TavilyWebSearch\", apiKey: \"{{ secret('TAVILY_API_KEY') }}\"}]"
    )
    @PluginProperty(group = "advanced")
    private Property<List<ContentRetrieverProvider>> contentRetrievers;

    @Schema(
        title = "Tools",
        description = "Tools the LLM may call to augment its answer. Not set by default (no tools).",
        example = "[{type: \"io.kestra.plugin.ai.tool.TavilyWebSearch\", apiKey: \"{{ secret('TAVILY_API_KEY') }}\"}]"
    )
    @PluginProperty(group = "destination")
    private List<ToolProvider> tools;

    @Schema(
        title = "Chat memory",
        description = "Store that persists the conversation history and replays it into the context on subsequent runs, so follow-up questions keep their context. Not set by default (each run starts fresh).",
        example = "{type: \"io.kestra.plugin.ai.memory.KestraKVStore\", memoryId: \"{{ inputs.session_id }}\"}"
    )
    @PluginProperty(group = "execution")
    private MemoryProvider memory;

    @Schema(
        title = "Guardrails",
        description = "Rules validating the call: input guardrails run against the user prompt before the LLM is called, output guardrails against the response before it is returned. The first failing rule stops execution and sets `guardrailViolated` to `true` in the output. Not set by default.",
        example = "{input: [{type: \"io.kestra.plugin.ai.guardrail.ExpressionInputGuardrail\", expression: \"{{ prompt | length < 5000 }}\"}]}"
    )
    @Nullable
    @PluginProperty(group = "advanced")
    private Guardrails guardrails;

    @Override
    public Output run(RunContext runContext) throws Exception {
        List<ToolProvider> toolProviders = ListUtils.emptyOnNull(tools);
        Duration taskTimeout = runContext.render(this.getTimeout()).as(Duration.class).orElse(Duration.ofSeconds(120));

        try {
            var chatModel = TokenBudgetChatModel.wrap(
                chatProvider.chatModel(runContext, chatConfiguration, taskTimeout),
                runContext,
                chatConfiguration
            );
            runContext.metric(Counter.of("ai.provider.calls", 1, "provider", chatProvider.getClass().getName()));
            AiServices<Assistant> assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(buildRetrievalAugmentor(runContext))
                .tools(AIUtils.buildTools(runContext, Collections.emptyMap(), toolProviders))
                .systemMessageProvider(throwFunction(memoryId -> runContext.render(systemMessage).as(String.class).orElse(null)))
                .toolArgumentsErrorHandler((error, context) ->
                {
                    runContext.logger().error(
                        "An error occurred while processing tool arguments for tool {} with request ID {}", context.toolExecutionRequest().name(), context.toolExecutionRequest().id(), error
                    );
                    throw new ToolArgumentsException(error);
                })
                .toolExecutionErrorHandler((error, context) ->
                {
                    runContext.logger()
                        .error("An error occurred during tool execution for tool {} with request ID {}", context.toolExecutionRequest().name(), context.toolExecutionRequest().id(), error);
                    throw new ToolExecutionException(error);
                });

            if (memory != null) {
                assistant.chatMemory(memory.chatMemory(runContext));
            }

            GuardrailsEvaluator.applyGuardrails(guardrails, assistant, runContext);
            String renderedPrompt = runContext.render(prompt).as(String.class).orElseThrow();
            //fallback timer in case no response is returned
            long fallbackStart = System.nanoTime();
            Result<AiMessage> completion = assistant.build().chat(renderedPrompt);
            long fallbackDuration = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - fallbackStart);
            runContext.logger().debug("Generated completion: {}", completion.content());

            // send metrics for token usage
            TokenUsage tokenUsage = TokenUsage.from(completion.tokenUsage());
            AIUtils.sendMetrics(runContext, tokenUsage);

            AIOutput output = AIOutput.from(runContext, completion, chatConfiguration.computeResponseFormat(runContext).type());
            Long requestDuration = output.getRequestDuration() == null ? fallbackDuration : output.getRequestDuration();
            return Output.builder()
                .completion(output.getTextOutput())
                .tokenUsage(output.getTokenUsage())
                .textOutput(output.getTextOutput())
                .jsonOutput(output.getJsonOutput())
                .finishReason(output.getFinishReason())
                .toolExecutions(output.getToolExecutions())
                .intermediateResponses(output.getIntermediateResponses())
                .requestDuration(requestDuration)
                .sources(output.getSources())
                .build();
        } catch (final InputGuardrailException | OutputGuardrailException e) {
            return buildGuardrailViolationOutput(runContext, e);
        } finally {
            toolProviders.forEach(tool -> tool.close(runContext));

            if (memory != null) {
                memory.close(runContext);
            }

            TimingChatModelListener.clear();
        }
    }

    private static Output buildGuardrailViolationOutput(RunContext runContext, GuardrailException e) {
        return Output.builder()
            .guardrailViolated(true)
            .guardrailViolationMessage(GuardrailsEvaluator.logAndFormatViolation(e, runContext.logger()))
            .build();
    }

    private RetrievalAugmentor buildRetrievalAugmentor(final RunContext runContext) throws Exception {
        List<ContentRetriever> toolContentRetrievers = runContext.render(contentRetrievers).asList(ContentRetrieverProvider.class).stream()
            .map(throwFunction(provider -> provider.contentRetriever(runContext)))
            .collect(Collectors.toList());

        Optional<ContentRetriever> contentRetriever = Optional.ofNullable(embeddings).map(
            throwFunction(
                embeddings ->
                {
                    var actualProvider = Optional.ofNullable(embeddingProvider).orElse(chatProvider);
                    var embeddingModel = actualProvider.embeddingModel(runContext);
                    runContext.metric(Counter.of("ai.provider.calls", 1, "provider", actualProvider.getClass().getName()));
                    var embeddingStore = embeddings.embeddingStore(runContext, embeddingModel.dimension(), false);
                    runContext.metric(Counter.of("ai.embedding.store.calls", 1, "store", embeddings.getClass().getName()));
                    return EmbeddingStoreContentRetriever.builder()
                        .embeddingModel(embeddingModel)
                        .embeddingStore(embeddingStore)
                        .maxResults(contentRetrieverConfiguration.getMaxResults())
                        .minScore(contentRetrieverConfiguration.getMinScore())
                        .build();
                }
            )
        );

        if (toolContentRetrievers.isEmpty() && contentRetriever.isEmpty()) {
            throw new IllegalArgumentException("Either `embeddings` or `contentRetrievers` must be provided.");
        }

        if (toolContentRetrievers.isEmpty()) {
            return DefaultRetrievalAugmentor.builder().contentRetriever(contentRetriever.get()).build();
        } else {
            // always add it first so it has precedence over the additional content retrievers
            contentRetriever.ifPresent(ct -> toolContentRetrievers.addFirst(ct));
            QueryRouter queryRouter = new DefaultQueryRouter(toolContentRetrievers.toArray(new ContentRetriever[0]));

            // Create a query router that will route each query to the embedding store content retriever and the tools content retrievers
            return DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();
        }
    }

    @Override
    public void kill() {
        if (this.tools != null) {
            this.tools.forEach(tool ->
            {
                try {
                    tool.kill();
                } catch (Exception ignored) {
                }
            });
        }
    }

    interface Assistant {
        Result<AiMessage> chat(String userMessage);
    }

    @Builder
    @Getter
    public static class ContentRetrieverConfiguration {
        @Schema(
            title = "Maximum results",
            description = "Number of matching segments the embedding store returns for the prompt. Defaults to `3`.",
            example = "5"
        )
        @Builder.Default
        private Integer maxResults = 3;

        @Schema(
            title = "Minimum similarity score",
            description = "Similarity threshold a match must reach to be injected into the context, from `0.0` (keep everything) to `1.0` (exact match only). Defaults to `0.0`.",
            example = "0.7"
        )
        @Builder.Default
        private Double minScore = 0.0D;
    }

    @SuperBuilder
    @Getter
    public static class Output extends AIOutput { // we must keep this one to keep the deprecated aiResponse
        @Schema(
            title = "Generated text completion",
            description = "Answer generated by the model. Deprecated: use `textOutput` or `jsonOutput` instead.",
            example = "Based on the retrieved documents, the refund window is 30 days."
        )
        @Deprecated(forRemoval = true, since = "1.0.0")
        private String completion;
    }
}
