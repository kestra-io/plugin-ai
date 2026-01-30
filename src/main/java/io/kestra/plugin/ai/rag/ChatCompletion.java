package io.kestra.plugin.ai.rag;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
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
import io.kestra.plugin.ai.domain.*;
import io.kestra.plugin.ai.provider.TimingChatModelListener;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

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
                      modelName: gemini-embedding-exp-03-07
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
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
                        messages:
                          - type: USER
                            content: Which features were released in Kestra 0.24?

                      - id: chat_with_rag
                        type: io.kestra.plugin.ai.rag.ChatCompletion
                        chatProvider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                        embeddingProvider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-embedding-exp-03-07
                        embeddings:
                          type: io.kestra.plugin.ai.embeddings.KestraKVStore
                        systemMessage: You are a helpful assistant that can answer questions about Kestra.
                        prompt: Which features were released in Kestra 0.24?

                pluginDefaults:
                  - type: io.kestra.plugin.ai.provider.GoogleGemini
                    values:
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash"""
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
                      modelName: gemini-2.5-flash
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.TavilyWebSearch
                        apiKey: "{{ kv('TAVILY_API_KEY') }}"
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
                    embeddingProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-exp-03-07
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    memory:
                      type: io.kestra.plugin.ai.memory.KestraKVStore
                      ttl: PT1M
                    systemMessage: You are a helpful assistant, answer concisely
                    prompt: "{{inputs.first}}"

                  - id: second
                    type: io.kestra.plugin.ai.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                    embeddingProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-exp-03-07
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    memory:
                      type: io.kestra.plugin.ai.memory.KestraKVStore
                    systemMessage: You are a helpful assistant, answer concisely
                    prompt: "{{inputs.second}}"

                pluginDefaults:
                  - type: io.kestra.plugin.ai.provider.GoogleGemini
                    values:
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash"""
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
                      apiKey: "{{ kv('MISTRAL_API_KEY') }}"
                      modelName: open-mistral-7b

                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.TavilyWebSearch
                        apiKey: "{{ kv('TAVILY_API_KEY') }}"
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
        )
    },
    aliases = "io.kestra.plugin.langchain4j.rag.ChatCompletion"
)
public class ChatCompletion extends Task implements RunnableTask<ChatCompletion.Output> {

    @Schema(title = "System message", description = "Instruction that sets the assistant's role, tone, and constraints for this task.")
    protected Property<String> systemMessage;

    @Schema(title = "User prompt", description = "The user input for this run. May be templated from flow inputs.")
    @NotNull
    protected Property<String> prompt;

    @Schema(
        title = "Embedding store",
        description = "Optional when at least one entry is provided in `contentRetrievers`."
    )
    @PluginProperty
    private EmbeddingStoreProvider embeddings;

    @Schema(
        title = "Embedding model provider",
        description = "Optional. If not set, the embedding model is created from `chatProvider`. Ensure the chosen chat provider supports embeddings."
    )
    @PluginProperty
    private ModelProvider embeddingProvider;

    @Schema(title = "Chat model provider")
    @NotNull
    @PluginProperty
    private ModelProvider chatProvider;

    @Schema(title = "Chat configuration")
    @NotNull
    @PluginProperty
    @Builder.Default
    private ChatConfiguration chatConfiguration = ChatConfiguration.empty();

    @Schema(title = "Content retriever configuration")
    @NotNull
    @PluginProperty
    @Builder.Default
    private ContentRetrieverConfiguration contentRetrieverConfiguration = ContentRetrieverConfiguration.builder().build();

    @Schema(
        title = "Additional content retrievers",
        description = "Some content retrievers like WebSearch can also be used as tools, but using them as content retrievers will ensure that they are always called whereas tools are only used when the LLM decides to."
    )
    private Property<List<ContentRetrieverProvider>> contentRetrievers;

    @Schema(title = "Optional tools the LLM may call to augment its response")
    private List<ToolProvider> tools;

    @Schema(
        title = "Chat memory",
        description = "Stores conversation history and injects it into context on subsequent runs."
    )
    private MemoryProvider memory;

    @Override
    public Output run(RunContext runContext) throws Exception {
        List<ToolProvider> toolProviders = ListUtils.emptyOnNull(tools);

        try {
            AiServices<Assistant> assistant = AiServices.builder(Assistant.class)
                .chatModel(chatProvider.chatModel(runContext, chatConfiguration))
                .retrievalAugmentor(buildRetrievalAugmentor(runContext))
                .tools(AIUtils.buildTools(runContext, Collections.emptyMap(), toolProviders))
                .systemMessageProvider(throwFunction(memoryId -> runContext.render(systemMessage).as(String.class).orElse(null)))
                .toolArgumentsErrorHandler((error, context) -> {
                    runContext.logger().error("An error occurred while processing tool arguments for tool {} with request ID {}", context.toolExecutionRequest().name(), context.toolExecutionRequest().id(), error);
                    throw new ToolArgumentsException(error);
                })
                .toolExecutionErrorHandler((error, context) -> {
                    runContext.logger().error("An error occurred during tool execution for tool {} with request ID {}", context.toolExecutionRequest().name(), context.toolExecutionRequest().id(), error);
                    throw new ToolExecutionException(error);
                });

            if (memory != null) {
                assistant.chatMemory(memory.chatMemory(runContext));
            }

            String renderedPrompt = runContext.render(prompt).as(String.class).orElseThrow();
            Result<AiMessage> completion = assistant.build().chat(renderedPrompt);
            runContext.logger().debug("Generated completion: {}", completion.content());

            // send metrics for token usage
            TokenUsage tokenUsage = TokenUsage.from(completion.tokenUsage());
            AIUtils.sendMetrics(runContext, tokenUsage);

            AIOutput output = AIOutput.from(runContext, completion, chatConfiguration.computeResponseFormat(runContext).type());
            return Output.builder()
                .completion(output.getTextOutput())
                .tokenUsage(output.getTokenUsage())
                .textOutput(output.getTextOutput())
                .jsonOutput(output.getJsonOutput())
                .finishReason(output.getFinishReason())
                .toolExecutions(output.getToolExecutions())
                .intermediateResponses(output.getIntermediateResponses())
                .requestDuration(output.getRequestDuration())
                .sources(output.getSources())
                .build();
        } finally {
            toolProviders.forEach(tool -> tool.close(runContext));

            if (memory != null) {
                memory.close(runContext);
            }

            TimingChatModelListener.clear();
        }
    }

    private RetrievalAugmentor buildRetrievalAugmentor(final RunContext runContext) throws Exception {
        List<ContentRetriever> toolContentRetrievers = runContext.render(contentRetrievers).asList(ContentRetrieverProvider.class).stream()
            .map(throwFunction(provider -> provider.contentRetriever(runContext)))
            .collect(Collectors.toList());

        Optional<ContentRetriever> contentRetriever = Optional.ofNullable(embeddings).map(throwFunction(
            embeddings -> {
                var embeddingModel = Optional.ofNullable(embeddingProvider).orElse(chatProvider).embeddingModel(runContext);
                return EmbeddingStoreContentRetriever.builder()
                    .embeddingModel(embeddingModel)
                    .embeddingStore(embeddings.embeddingStore(runContext, embeddingModel.dimension(), false))
                    .maxResults(contentRetrieverConfiguration.getMaxResults())
                    .minScore(contentRetrieverConfiguration.getMinScore())
                    .build();
            }));

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
            this.tools.forEach(tool -> {
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
        @Schema(title = "Maximum results to return from the embedding store")
        @Builder.Default
        private Integer maxResults = 3;

        @Schema(title = "Minimum similarity score (0-1 inclusive). Only results with score ≥ minScore are returned.")
        @Builder.Default
        private Double minScore = 0.0D;
    }

    @SuperBuilder
    @Getter
    public static class Output extends AIOutput { // we must keep this one to keep the deprecated aiResponse
        @Schema(title = "Generated text completion", description = "Deprecated. Use `textOutput` or `jsonOutput` instead.")
        @Deprecated(forRemoval = true, since = "1.0.0")
        private String completion;
    }
}
