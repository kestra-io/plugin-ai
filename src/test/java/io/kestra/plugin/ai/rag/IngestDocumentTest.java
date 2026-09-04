package io.kestra.plugin.ai.rag;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.junit.jupiter.api.parallel.ResourceLock;

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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@Execution(ExecutionMode.SAME_THREAD)
@ResourceLock("kestra-h2-flyway")
@KestraTest
class IngestDocumentTest extends ContainerTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void inlineDocuments() throws Exception {
        RunContext runContext = ollamaRunContext();

        var task = ingestTaskBuilder()
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
        RunContext runContext = ollamaRunContext();

        Path path = runContext.workingDir().createFile("document.txt");
        Files.write(path, "I'm Loïc".getBytes());
        URI uri = runContext.storage().putFile(path.toFile());

        var task = ingestTaskBuilder()
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
        RunContext runContext = ollamaRunContext();

        Path path1 = runContext.workingDir().createFile("ingest/document1.txt");
        Files.write(path1, "I'm Loïc".getBytes());
        Path path2 = runContext.workingDir().createFile("ingest/document2.txt");
        Files.write(path2, "I live in Lille".getBytes());

        var task = ingestTaskBuilder()
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
        RunContext runContext = ollamaRunContext();

        var task = ingestTaskBuilder()
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
        RunContext runContext = ollamaRunContext();

        int bulkSize = 5;
        int totalDocs = 12;

        List<IngestDocument.InlineDocument> docs = IntStream.range(0, totalDocs)
            .mapToObj(
                i -> IngestDocument.InlineDocument.builder()
                    .content(Property.ofValue("doc-" + i))
                    .build()
            )
            .toList();

        var task = ingestTaskBuilder()
            .bulkSize(Property.ofValue(bulkSize))
            .fromDocuments(docs)
            .build();

        IngestDocument.Output output = task.run(runContext);

        assertThat(output.getIngestedDocuments()).isEqualTo(totalDocs);

        String kvKey = (String) output.getEmbeddingStoreOutputs().get("kvName");
        KVStore kvStore = runContext.namespaceKv(runContext.flowInfo().namespace());
        assertKvStore(kvStore, kvKey, totalDocs);
    }

    @Test
    void topLevelMetadataFromInternalURIs() throws Exception {
        RunContext runContext = ollamaRunContext();

        Path path = runContext.workingDir().createFile("document.txt");
        Files.write(path, "I'm Loïc".getBytes());
        URI uri = runContext.storage().putFile(path.toFile());

        var task = ingestTaskBuilder()
            .metadata(Property.ofValue(Map.of("source", "manual-run", "team", "data")))
            .fromInternalURIs(Property.ofValue(List.of(uri.toString())))
            .build();

        IngestDocument.Output output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);

        List<Map<String, Object>> metadata = storedMetadata(runContext, output);
        assertThat(metadata).hasSize(1);
        assertThat(metadata.getFirst()).containsEntry("source", "manual-run").containsEntry("team", "data");
    }

    @Test
    void topLevelMetadataFromExternalURLs() throws Exception {
        RunContext runContext = ollamaRunContext();

        // a file: URL keeps this test offline while still going through UrlDocumentLoader, which injects the `url` metadata key
        Path path = runContext.workingDir().createFile("external/document.txt");
        Files.writeString(path, "I'm Loïc");
        String url = path.toUri().toURL().toString();

        var task = ingestTaskBuilder()
            .metadata(Property.ofValue(Map.of("source", "manual-run", "url", "should-not-win")))
            .fromExternalURLs(Property.ofValue(List.of(url)))
            .build();

        IngestDocument.Output output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);

        List<Map<String, Object>> metadata = storedMetadata(runContext, output);
        assertThat(metadata).hasSize(1);
        assertThat(metadata.getFirst())
            .containsEntry("source", "manual-run")
            .containsEntry("url", url);
    }

    @Test
    void topLevelMetadataFromPathDoesNotOverwriteLoaderMetadata() throws Exception {
        RunContext runContext = ollamaRunContext();

        Path path = runContext.workingDir().createFile("ingest/document1.txt");
        Files.write(path, "I'm Loïc".getBytes());

        var task = ingestTaskBuilder()
            .metadata(Property.ofValue(Map.of("source", "manual-run", "file_name", "should-not-win.txt")))
            .fromPath(Property.ofValue("ingest"))
            .build();

        IngestDocument.Output output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);

        List<Map<String, Object>> metadata = storedMetadata(runContext, output);
        assertThat(metadata).hasSize(1);
        assertThat(metadata.getFirst())
            .containsEntry("source", "manual-run")
            .containsEntry("file_name", "document1.txt");
    }

    @Test
    void inlineDocumentMetadataTakesPrecedenceOverTopLevelMetadata() throws Exception {
        RunContext runContext = ollamaRunContext();

        var task = ingestTaskBuilder()
            .metadata(Property.ofValue(Map.of("source", "manual-run", "team", "data")))
            .fromDocuments(
                List.of(
                    IngestDocument.InlineDocument.builder()
                        .content(Property.ofValue("I'm Loïc"))
                        .metadata(Property.ofValue(Map.of("team", "platform")))
                        .build()
                )
            )
            .build();

        IngestDocument.Output output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);

        List<Map<String, Object>> metadata = storedMetadata(runContext, output);
        assertThat(metadata).hasSize(1);
        assertThat(metadata.getFirst())
            .containsEntry("source", "manual-run")
            .containsEntry("team", "platform");
    }

    @Test
    void withoutTopLevelMetadataDocumentsKeepTheirOwnMetadata() throws Exception {
        RunContext runContext = ollamaRunContext();

        var task = ingestTaskBuilder()
            .fromDocuments(
                List.of(
                    IngestDocument.InlineDocument.builder()
                        .content(Property.ofValue("I'm Loïc"))
                        .metadata(Property.ofValue(Map.of("team", "platform")))
                        .build()
                )
            )
            .build();

        IngestDocument.Output output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);

        List<Map<String, Object>> metadata = storedMetadata(runContext, output);
        assertThat(metadata).hasSize(1);
        assertThat(metadata.getFirst()).containsEntry("team", "platform").doesNotContainKey("source");
    }

    @Test
    void unsupportedMetadataValueTypeFailsBeforeIngestion() {
        RunContext runContext = ollamaRunContext();

        var task = ingestTaskBuilder()
            .metadata(Property.ofValue(Map.of("tags", List.of("a", "b"))))
            .fromDocuments(List.of(IngestDocument.InlineDocument.builder().content(Property.ofValue("I'm Loïc")).build()))
            .build();

        assertThatThrownBy(() -> task.run(runContext))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("tags")
            .hasMessageContaining("String, UUID, Integer, Long, Float or Double");
    }

    private RunContext ollamaRunContext() {
        return runContextFactory.of(
            "namespace", Map.of(
                "modelName", "chroma/all-minilm-l6-v2-f32",
                "endpoint", ollamaEndpoint
            )
        );
    }

    private IngestDocument.IngestDocumentBuilder<?, ?> ingestTaskBuilder() {
        return IngestDocument.builder()
            .provider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(KestraKVStore.builder().build())
            .drop(Property.ofValue(true));
    }

    private List<Map<String, Object>> storedMetadata(RunContext runContext, IngestDocument.Output output) throws IOException, ResourceExpiredException {
        String kvKey = (String) output.getEmbeddingStoreOutputs().get("kvName");
        KVStore kvStore = runContext.namespaceKv(runContext.flowInfo().namespace());
        Optional<KVEntry> kvEntry = kvStore.get(kvKey);
        assertThat(kvEntry.isPresent()).isTrue();
        Optional<KVValue> kvValue = kvStore.getValue(kvEntry.get().key());
        JsonNode jsonNode = JacksonMapper.ofJson().readTree(kvValue.orElseThrow().value().toString());

        List<Map<String, Object>> metadata = new ArrayList<>();
        for (JsonNode entry : jsonNode.get("entries")) {
            JsonNode node = entry.get("embedded").get("metadata");
            if (node.has("metadata")) {
                node = node.get("metadata");
            }
            metadata.add(JacksonMapper.ofJson().convertValue(node, Map.class));
        }
        return metadata;
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