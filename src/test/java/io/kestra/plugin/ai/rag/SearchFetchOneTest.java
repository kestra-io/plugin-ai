package io.kestra.plugin.ai.rag;

import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.common.FetchType;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;
import io.kestra.plugin.ai.provider.Ollama;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class SearchFetchOneTest {
    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void returnsEmptyResultsWhenNoMatchIsFound() throws Exception {
        var provider = fixedEmbeddingProvider(1.0f);

        var task = Search.builder()
            .provider(provider)
            .embeddings(new ClosableEmbeddingStoreProvider())
            .query(Property.ofValue("Banana"))
            .maxResults(Property.ofValue(5))
            .minScore(Property.ofValue(0.8))
            .fetchType(Property.ofValue(FetchType.FETCH_ONE))
            .build();

        var output = task.run(runContextFactory.of("namespace", Map.of()));

        assertThat(output.getResults()).isEmpty();
        assertThat(output.getSize()).isZero();
    }

    @Test
    void returnsOnlyTheBestMatchWithTheTotalSize() throws Exception {
        var store = new InMemoryEmbeddingStore<TextSegment>();
        store.add(Embedding.from(new float[] { 1.0f, 1.0f }), TextSegment.from("Apple"));
        store.add(Embedding.from(new float[] { 1.0f, 0.0f }), TextSegment.from("Banana"));

        var provider = fixedEmbeddingProvider(1.0f, 0.0f);

        var embeddings = new EmbeddingStoreProvider() {
            @Override
            public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) {
                return store;
            }
        };

        var task = Search.builder()
            .provider(provider)
            .embeddings(embeddings)
            .query(Property.ofValue("Banana"))
            .maxResults(Property.ofValue(5))
            .minScore(Property.ofValue(0.8))
            .fetchType(Property.ofValue(FetchType.FETCH_ONE))
            .build();

        var output = task.run(runContextFactory.of("namespace", Map.of()));

        assertThat(output.getResults()).containsExactly("Banana");
        assertThat(output.getSize()).isEqualTo(2);
    }

    private static Ollama fixedEmbeddingProvider(float... vector) {
        return new Ollama() {
            @Override
            public EmbeddingModel embeddingModel(RunContext runContext) {
                return new EmbeddingModel() {
                    @Override
                    public Response<Embedding> embed(String text) {
                        return Response.from(Embedding.from(vector));
                    }

                    @Override
                    public int dimension() {
                        return vector.length;
                    }
                };
            }
        };
    }
}
