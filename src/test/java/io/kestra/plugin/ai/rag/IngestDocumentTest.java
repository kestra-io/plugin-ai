package io.kestra.plugin.ai.rag;

import com.fasterxml.jackson.databind.JsonNode;
import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.exceptions.ResourceExpiredException;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.core.storages.kv.KVEntry;
import io.kestra.core.storages.kv.KVStore;
import io.kestra.core.storages.kv.KVValue;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.embeddings.KestraKVStore;
import io.kestra.plugin.ai.provider.Ollama;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class IngestDocumentTest extends ContainerTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void inlineDocuments() throws Exception {
        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "tinydolphin",
            "endpoint", ollamaEndpoint
        ));

        var task = IngestDocument.builder()
            .provider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(
                KestraKVStore.builder().build()
            )
            .drop(Property.ofValue(true))
            .fromDocuments(List.of(IngestDocument.InlineDocument.builder().content(Property.ofValue("I'm Loïc")).build()))
            .build();

        IngestDocument.Output output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);

        String kvKey = (String) output.getEmbeddingStoreOutputs().get("kvName");
        KVStore kvStore = runContext.namespaceKv(runContext.flowInfo().namespace());
        assertKvStore(kvStore, kvKey, 1);
    }

    @Test
    void internalStorageURIs() throws Exception {
        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "tinydolphin",
            "endpoint", ollamaEndpoint
        ));

        Path path = runContext.workingDir().createFile("document.txt");
        Files.write(path, "I'm Loïc".getBytes());
        URI uri = runContext.storage().putFile(path.toFile());

        var task = IngestDocument.builder()
            .provider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(
                KestraKVStore.builder().build()
            )
            .drop(Property.ofValue(true))
            .fromInternalURIs(Property.ofValue(List.of(uri.toString())))
            .build();

        IngestDocument.Output output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);

        String kvKey = (String) output.getEmbeddingStoreOutputs().get("kvName");
        KVStore kvStore = runContext.namespaceKv(runContext.flowInfo().namespace());
        assertKvStore(kvStore, kvKey, 1);
    }

    @Test
    void workingDirectoryPath() throws Exception {
        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "tinydolphin",
            "endpoint", ollamaEndpoint
        ));

        Path path1 = runContext.workingDir().createFile("ingest/document1.txt");
        Files.write(path1, "I'm Loïc".getBytes());
        Path path2 = runContext.workingDir().createFile("ingest/document2.txt");
        Files.write(path2, "I live in Lille".getBytes());

        var task = IngestDocument.builder()
            .provider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(
                KestraKVStore.builder().build()
            )
            .drop(Property.ofValue(true))
            .fromPath(Property.ofValue("ingest"))
            .build();

        IngestDocument.Output output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(2);

        String kvKey = (String) output.getEmbeddingStoreOutputs().get("kvName");
        KVStore kvStore = runContext.namespaceKv(runContext.flowInfo().namespace());
        assertKvStore(kvStore, kvKey, 2);
    }

    @Test
    void externalURLs() throws Exception {
        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "tinydolphin",
            "endpoint", ollamaEndpoint
        ));

        var task = IngestDocument.builder()
            .provider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(
                KestraKVStore.builder().build()
            )
            .drop(Property.ofValue(true))
            .fromExternalURLs(Property.ofValue(List.of("https://dummyjson.com/products/1", "https://dummyjson.com/products/2")))
            .build();

        IngestDocument.Output output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(2);

        String kvKey = (String) output.getEmbeddingStoreOutputs().get("kvName");
        KVStore kvStore = runContext.namespaceKv(runContext.flowInfo().namespace());
        assertKvStore(kvStore, kvKey, 2);
    }

    @Test
    void inlineDocumentsWithBulkSize() throws Exception {
        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "tinydolphin",
            "endpoint", ollamaEndpoint
        ));

        int bulkSize = 5;
        int totalDocs = 12;

        List<IngestDocument.InlineDocument> docs = IntStream.range(0, totalDocs)
                .mapToObj(i -> IngestDocument.InlineDocument.builder()
                    .content(Property.ofValue("doc-" + i))
                    .build()
                )
                .toList();

        var task = IngestDocument.builder()
            .provider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(KestraKVStore.builder().build())
            .drop(Property.ofValue(true))
            .bulkSize(Property.ofValue(bulkSize))
            .fromDocuments(docs)
            .build();

        IngestDocument.Output output = task.run(runContext);

        assertThat(output.getIngestedDocuments()).isEqualTo(totalDocs);

        String kvKey = (String) output.getEmbeddingStoreOutputs().get("kvName");
        KVStore kvStore = runContext.namespaceKv(runContext.flowInfo().namespace());
        assertKvStore(kvStore, kvKey, totalDocs);
    }


    private void assertKvStore(KVStore kvStore, String kvKey, int nbDocuments) throws IOException, ResourceExpiredException {
        Optional<KVEntry> kvEntry = kvStore.get(kvKey);
        assertThat(kvEntry.isPresent()).isTrue();
        Optional<KVValue> kvValue = kvStore.getValue(kvEntry.get().key());
        assertThat(kvValue.isPresent()).isTrue();
        assertThat(kvValue.get().value()).isNotNull();
        String value = kvValue.get().value().toString();
        JsonNode jsonNode = JacksonMapper.ofJson().readTree(value);
        assertThat(jsonNode.get("entries")).isNotNull();
        assertThat(jsonNode.get("entries").size()).isEqualTo(nbDocuments);
    }
}