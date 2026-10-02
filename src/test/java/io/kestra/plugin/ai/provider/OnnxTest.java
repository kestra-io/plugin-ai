package io.kestra.plugin.ai.provider;

import java.io.InputStream;
import java.net.URI;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.junit.jupiter.api.parallel.ResourceLock;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.common.FetchType;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.embeddings.KestraKVStore;
import io.kestra.plugin.ai.rag.IngestDocument;
import io.kestra.plugin.ai.rag.Search;

import dev.langchain4j.store.embedding.CosineSimilarity;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

// Model files come from the langchain4j-embeddings-all-minilm-l6-v2 test dependency: no network, no container.
@Execution(ExecutionMode.SAME_THREAD)
@ResourceLock("kestra-h2-flyway")
@KestraTest
class OnnxTest {
    private static final String MODEL_RESOURCE = "all-minilm-l6-v2.onnx";
    private static final String TOKENIZER_RESOURCE = "all-minilm-l6-v2-tokenizer.json";

    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void embedsFromInternalStorageAndReusesModelForSameFilesFromNamespace() throws Exception {
        var runContext = runContextFactory.of("company.ai", Map.of());

        var fromStorage = provider(
            putInStorage(runContext, MODEL_RESOURCE).toString(),
            putInStorage(runContext, TOKENIZER_RESOURCE).toString()
        );
        var embeddingModel = fromStorage.embeddingModel(runContext);

        assertThat(embeddingModel.dimension()).isEqualTo(384);
        var kestra = embeddingModel.embed("Kestra orchestrates data pipelines").content();
        var orchestration = embeddingModel.embed("Kestra is a workflow orchestration tool").content();
        var cat = embeddingModel.embed("My cat likes sleeping in the sun").content();
        assertThat(CosineSimilarity.between(kestra, orchestration))
            .isGreaterThan(CosineSimilarity.between(kestra, cat) + 0.3);

        putInNamespace(runContext, MODEL_RESOURCE, "models/minilm/model.onnx");
        putInNamespace(runContext, TOKENIZER_RESOURCE, "models/minilm/tokenizer.json");
        var fromNamespace = provider("nsfile:///models/minilm/model.onnx", "nsfile:///models/minilm/tokenizer.json");

        // same file content behind different URIs must not load a second native session
        assertThat(fromNamespace.embeddingModel(runContext)).isSameAs(embeddingModel);
    }

    @Test
    void ingestsAndSearchesWithKestraKVStore() throws Exception {
        var runContext = runContextFactory.of("company.ai", Map.of());
        var onnx = provider(
            putInStorage(runContext, MODEL_RESOURCE).toString(),
            putInStorage(runContext, TOKENIZER_RESOURCE).toString()
        );
        var store = KestraKVStore.builder().build();

        var ingested = IngestDocument.builder()
            .provider(onnx)
            .embeddings(store)
            .drop(Property.ofValue(true))
            .fromDocuments(
                List.of(
                    IngestDocument.InlineDocument.builder().content(Property.ofValue("Kestra is an open-source orchestration platform for data and AI workflows.")).build(),
                    IngestDocument.InlineDocument.builder().content(Property.ofValue("Bananas are a good source of potassium.")).build(),
                    IngestDocument.InlineDocument.builder().content(Property.ofValue("PostgreSQL is a relational database.")).build()
                )
            )
            .build()
            .run(runContext);
        assertThat(ingested.getIngestedDocuments()).isEqualTo(3);

        var found = Search.builder()
            .provider(onnx)
            .embeddings(store)
            .query(Property.ofValue("Which tool orchestrates workflows?"))
            .maxResults(Property.ofValue(1))
            .minScore(Property.ofValue(0.3))
            .fetchType(Property.ofValue(FetchType.FETCH))
            .build()
            .run(runContext);

        assertThat(found.getResults()).isEqualTo(List.of("Kestra is an open-source orchestration platform for data and AI workflows."));
    }

    private static Onnx provider(String modelUri, String tokenizerUri) {
        return Onnx.builder()
            .type(Onnx.class.getName())
            .modelName(Property.ofValue("all-MiniLM-L6-v2"))
            .modelUri(Property.ofValue(modelUri))
            .tokenizerUri(Property.ofValue(tokenizerUri))
            .build();
    }

    private static URI putInStorage(RunContext runContext, String resource) throws Exception {
        try (InputStream in = resource(resource)) {
            return runContext.storage().putFile(in, resource);
        }
    }

    private static void putInNamespace(RunContext runContext, String resource, String path) throws Exception {
        try (InputStream in = resource(resource)) {
            runContext.storage().namespace().putFile(Path.of(path), in);
        }
    }

    private static InputStream resource(String name) {
        return OnnxTest.class.getClassLoader().getResourceAsStream(name);
    }
}
