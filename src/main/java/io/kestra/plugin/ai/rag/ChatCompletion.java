package io.kestra.plugin.ai.rag;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
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
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
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
@Schema(title = "Create a Retrieval Augmented Generation (RAG) pipeline")
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Chat with your data using Retrieval Augmented Generation (RAG)",
            code = """
                id: rag_chat
                namespace: company.ai

                tasks:
                  - id: chat_with_rag
                    type: io.kestra.plugin.ai.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.0-flash
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                    embeddingProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-exp-03-07
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    messages:
                      - type: SYSTEM
                        content: You are a helpful assistant that can answer questions about Kestra releases.
                      - type: USER
                        content: Which features were released in Kestra 0.24?
                """
        )
    },
    aliases = "io.kestra.plugin.langchain4j.rag.ChatCompletion"
)
public class ChatCompletion extends Task implements RunnableTask<ChatCompletion.Output> {

    @Schema(
        title = "Chat Messages",
        description = "The list of chat messages for the conversation. There can be only one system message, and the last message must be a user message."
    )
    @NotNull
    private Property<List<io.kestra.plugin.ai.domain.ChatMessage>> messages;

    @Schema(
        title = "Embedding store",
        description = "Optional when at least one entry is provided in `contentRetrievers`."
    )
    @PluginProperty
    private EmbeddingStoreProvider embeddings;

    @Schema(
        title = "Embedding model provider",
        description = "Optional. If not set, the embedding model is created from `chatProvider`."
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
        description = "Optional web or file retrievers to augment the RAG response."
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
            List<io.kestra.plugin.ai.domain.ChatMessage> rMessages =
                runContext.render(messages).asList(io.kestra.plugin.ai.domain.ChatMessage.class);

            if (rMessages.isEmpty()) {
                throw new IllegalArgumentException("At least one user message must be provided.");
            }

            // Convert to LangChain4j messages
            List<dev.langchain4j.data.message.ChatMessage> chatMessages = rMessages.stream()
                .map(msg -> switch (msg.type()) {
                    case SYSTEM -> SystemMessage.systemMessage(msg.content());
                    case USER -> UserMessage.userMessage(msg.content());
                    case AI -> AiMessage.aiMessage(msg.content());
                })
                .toList();

            AiServices<Assistant> assistant = AiServices.builder(Assistant.class)
                .chatModel(chatProvider.chatModel(runContext, chatConfiguration))
                .retrievalAugmentor(buildRetrievalAugmentor(runContext))
                .tools(AIUtils.buildTools(runContext, Collections.emptyMap(), toolProviders))
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

            // Extract the last USER message to chat
            String userPrompt = chatMessages.getLast() instanceof UserMessage um ? um.singleText() : "";

            Result<AiMessage> completion = assistant.build().chat(userPrompt);
            runContext.logger().debug("Generated RAG completion: {}", completion.content());

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
            contentRetriever.ifPresent(ct -> toolContentRetrievers.addFirst(ct));
            QueryRouter queryRouter = new DefaultQueryRouter(toolContentRetrievers.toArray(new ContentRetriever[0]));
            return DefaultRetrievalAugmentor.builder().queryRouter(queryRouter).build();
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
    public static class Output extends AIOutput {
        @Schema(title = "Generated text completion", description = "Deprecated. Use `textOutput` or `jsonOutput` instead.")
        @Deprecated(forRemoval = true, since = "1.0.0")
        private String completion;
    }
}
