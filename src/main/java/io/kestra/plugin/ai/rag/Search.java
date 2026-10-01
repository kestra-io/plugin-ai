package io.kestra.plugin.ai.rag;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URI;
import java.util.AbstractMap;
import java.util.List;
import java.util.Map;

import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Metric;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.models.tasks.common.FetchType;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.FileSerde;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;
import io.kestra.plugin.ai.domain.ModelProvider;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;
import reactor.core.publisher.Flux;
import reactor.core.publisher.FluxSink;

import static io.kestra.core.models.tasks.common.FetchType.NONE;

@SuperBuilder
@Getter
@NoArgsConstructor
@ToString
@EqualsAndHashCode
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Search an embedding store",
            code = """
                id: search_embeddings_flow
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
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-22.md

                  - id: search
                    type: io.kestra.plugin.ai.rag.Search
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-001
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    query: "Feature Highlights"
                    maxResults: 5
                    minScore: 0.5
                    fetchType: FETCH
                """
        ),
    },
    metrics = {
        @Metric(
            name = "fetch.size",
            type = Counter.TYPE,
            unit = "records",
            description = "The number of rows fetch from the embedding store."
        ),
        @Metric(
            name = "ai.provider.calls",
            type = Counter.TYPE,
            unit = "calls",
            description = "Number of times an embedding model is obtained from a provider, tagged by provider class name"
        ),
        @Metric(
            name = "ai.embedding.store.calls",
            type = Counter.TYPE,
            unit = "calls",
            description = "Number of times an embedding store is used, tagged by store class name"
        )
    },
    aliases = "io.kestra.plugin.langchain4j.rag.Search"
)
@Schema(
    title = "Search embeddings and optionally return results",
    description = """
        Runs semantic search against the configured embedding store using the embedding of the provided `query`. `maxResults` limits hits and `minScore` filters low-similarity matches. `fetchType` controls output: NONE (metrics only), FETCH/FETCH_ONE (return matches), or STORE (write matches to internal storage)."""
)
public class Search extends Task implements RunnableTask<Search.Output> {

    @Schema(
        title = "Query string",
        description = "Text embedded and matched against the stored vectors. No default: this property is required.",
        example = "{{ inputs.question }}"
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<String> query;

    @Schema(
        title = "Maximum results",
        description = "Number of matching segments returned by the embedding store. No default: this property is required.",
        example = "5"
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<Integer> maxResults;

    @Schema(
        title = "Minimum similarity score",
        description = "Similarity threshold a match must reach to be returned, from `0.0` (keep everything) to `1.0` (exact match only). No default: this property is required.",
        example = "0.7"
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<Double> minScore;

    @Schema(
        title = "Embedding model provider",
        description = "Model provider used to embed the query. It should use the same embedding model that was used at ingestion time. No default: this property is required.",
        example = "{type: \"io.kestra.plugin.ai.provider.GoogleGemini\", apiKey: \"{{ secret('GEMINI_API_KEY') }}\", modelName: \"gemini-embedding-001\"}"
    )
    @NotNull
    @PluginProperty(group = "main")
    private ModelProvider provider;

    @Schema(
        title = "Embedding store provider",
        description = "Vector store searched for segments matching the query. No default: this property is required.",
        example = "{type: \"io.kestra.plugin.ai.embeddings.KestraKVStore\"}"
    )
    @NotNull
    @PluginProperty(group = "main")
    private EmbeddingStoreProvider embeddings;

    @Schema(
        title = "Fetch type",
        description = "What the task returns: `NONE` emits only the match count, `FETCH` returns all matches in `results`, `FETCH_ONE` returns the single best match, and `STORE` writes the matches to Kestra internal storage and emits their `uri`. Defaults to `NONE`.",
        example = "FETCH"
    )
    @NotNull
    @Builder.Default
    @PluginProperty(group = "processing")
    protected Property<FetchType> fetchType = Property.ofValue(NONE);

    @Override
    public Output run(RunContext runContext) throws Exception {
        var embeddingModel = provider.embeddingModel(runContext);
        runContext.metric(Counter.of("ai.provider.calls", 1, "provider", provider.getClass().getName()));
        var store = embeddings.embeddingStore(runContext, embeddingModel.dimension(), false);
        runContext.metric(Counter.of("ai.embedding.store.calls", 1, "store", embeddings.getClass().getName()));

        var renderedQuery = runContext.render(query).as(String.class).orElseThrow();
        var embedding = embeddingModel.embed(renderedQuery).content();

        var request = EmbeddingSearchRequest.builder()
            .queryEmbedding(embedding)
            .maxResults(runContext.render(maxResults).as(Integer.class).orElseThrow())
            .minScore(runContext.render(minScore).as(Double.class).orElseThrow())
            .build();

        var results = store.search(request).matches().stream()
            .map(EmbeddingMatch::embedded)
            .map(TextSegment::text)
            .toList();

        Output output;

        int fetchedItemsCount = results.size();
        var renderedFetchType = runContext.render(this.fetchType).as(FetchType.class).orElse(NONE);
        switch (renderedFetchType) {
            case NONE:
                output = Output.builder().build();
                runContext.metric(Counter.of("fetch.size", 0, "fetch", "false", "store", "false"));
                break;
            case FETCH:
                output = Output.builder()
                    .results(results)
                    .size(results.size())
                    .build();
                runContext.metric(Counter.of("fetch.size", fetchedItemsCount, "fetch", "true", "store", "false"));
                break;
            case FETCH_ONE:
                output = Output.builder()
                    .results(List.of(results.getFirst()))
                    .size(fetchedItemsCount)
                    .build();
                runContext.metric(Counter.of("fetch.size", fetchedItemsCount, "fetch", "true", "store", "false"));
                break;
            case STORE:
                var result = storeResult(results, runContext);
                int storedItemsCount = result.getValue().intValue();
                output = Output.builder()
                    .uri(result.getKey())
                    .size(storedItemsCount)
                    .build();
                runContext.metric(Counter.of("fetch.size", storedItemsCount, "fetch", "false", "store", "true"));
                break;
            default:
                throw new IllegalStateException("Unexpected fetchType value: " + fetchType);
        }

        return output;
    }

    private Map.Entry<URI, Long> storeResult(List<String> results, RunContext runContext) throws IOException {
        var tempFile = runContext.workingDir().createTempFile(".ion").toFile();

        try (
            var output = new BufferedWriter(new FileWriter(tempFile), FileSerde.BUFFER_SIZE)
        ) {
            var flowable = Flux
                .create(
                    s ->
                    {
                        results.forEach(s::next);
                        s.complete();
                    },
                    FluxSink.OverflowStrategy.BUFFER
                );

            var count = FileSerde.writeAll(output, flowable);
            var lineCount = count.block();

            output.flush();

            return new AbstractMap.SimpleEntry<>(
                runContext.storage().putFile(tempFile),
                lineCount
            );
        }
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {

        @Schema(
            title = "Matching text results",
            description = "Text segments matching the query, ordered by descending similarity. Populated only when `fetchType` is `FETCH` or `FETCH_ONE`.",
            example = "[\"Refunds are accepted within 30 days of purchase.\"]"
        )
        private final List<String> results;

        @Schema(
            title = "Output file URI",
            description = "Kestra internal storage URI of the ion file holding the matches. Available only when `fetchType` is `STORE`.",
            example = "kestra:///company/team/rag_search/executions/abc123/tasks/search/output.ion"
        )
        private final URI uri;

        @Schema(
            title = "Number of matches",
            description = "Total number of matches the search found. For `FETCH_ONE` this is still the total, not the single item returned in `results`.",
            example = "5"
        )
        private Integer size;
    }
}
