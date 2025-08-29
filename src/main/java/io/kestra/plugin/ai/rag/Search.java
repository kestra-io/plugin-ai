package io.kestra.plugin.ai.rag;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import io.kestra.core.models.annotations.Example;
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
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;
import reactor.core.publisher.Flux;
import reactor.core.publisher.FluxSink;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URI;
import java.util.AbstractMap;
import java.util.List;
import java.util.Map;

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
                      modelName: gemini-embedding-exp-03-07
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    drop: true
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-22.md

                  - id: search
                    type: io.kestra.plugin.ai.rag.Search
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-exp-03-07
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    query: "Feature Highlights"
                    maxResults: 5
                    minScore: 0.5
                    fetchType: FETCH
                """
        ),
    },
    aliases = "io.kestra.plugin.langchain4j.rag.Search"
)
@Schema(
    title = "Search an embedding store",
    description = "Perform a semantic search using a query string."
)
public class Search extends Task implements RunnableTask<Search.Output> {

    @Schema(title = "Query string")
    @NotNull
    private Property<String> query;

    @Schema(title = "Maximum number of results")
    @NotNull
    private Property<Integer> maxResults;

    @Schema(title = "Minimum similarity score")
    @NotNull
    private Property<Double> minScore;

    @Schema(title = "Embedding model provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @Schema(title = "Embedding store provider")
    @NotNull
    @PluginProperty
    private EmbeddingStoreProvider embeddings;

    @NotNull
    @Builder.Default
    protected Property<FetchType> fetchType = Property.ofValue(NONE);

    @Override
    public Output run(RunContext runContext) throws Exception {
        var embeddingModel = provider.embeddingModel(runContext);
        var store = embeddings.embeddingStore(runContext, embeddingModel.dimension(), false);

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
                runContext.metric(Counter.of("store.fetchedItemsCount", 0));
                runContext.metric(Counter.of("fetch.fetchedItemsCount", 0));
                break;
            case FETCH:
                output = Output.builder()
                    .results(results)
                    .size(results.size())
                    .build();
                runContext.metric(Counter.of("fetch.fetchedItemsCount", fetchedItemsCount));
                break;
            case FETCH_ONE:
                output = Output.builder()
                    .results(List.of(results.getFirst()))
                    .size(fetchedItemsCount)
                    .build();
                runContext.metric(Counter.of("fetch.fetchedItemsCount", fetchedItemsCount));
                break;
            case STORE:
                var result = storeResult(results, runContext);
                int storedItemsCount = result.getValue().intValue();
                output = Output.builder()
                    .uri(result.getKey())
                    .size(storedItemsCount)
                    .build();
                runContext.metric(Counter.of("store.fetchedItemsCount", storedItemsCount));
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
                    s -> {
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

        @Schema(title = "List of matching text results")
        private final List<String> results;

        @Schema(
            title = "Output file URI in Kestra’s internal storage",
            description = "Only available when `fetchType` is set to `STORE`"
        )
        private final URI uri;

        @Schema(title = "Count of fetched or stored items")
        private Integer size;
    }
}
