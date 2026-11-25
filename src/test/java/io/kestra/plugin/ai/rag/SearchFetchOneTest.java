package io.kestra.plugin.ai.rag;

import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.common.FetchType;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.provider.Ollama;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class SearchFetchOneTest {
    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void returnsEmptyResultsWhenNoMatchIsFound() throws Exception {
        var provider = new Ollama() {
            @Override
            public EmbeddingModel embeddingModel(RunContext runContext) {
                return new EmbeddingModel() {
                    @Override
                    public Response<Embedding> embed(String text) {
                        return Response.from(Embedding.from(new float[] { 1.0f }));
                    }

                    @Override
                    public int dimension() {
                        return 1;
                    }
                };
            }
        };

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
}
