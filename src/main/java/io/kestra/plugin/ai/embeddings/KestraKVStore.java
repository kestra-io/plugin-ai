package io.kestra.plugin.ai.embeddings;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.exceptions.ResourceExpiredException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.storages.kv.KVEntry;
import io.kestra.core.storages.kv.KVValue;
import io.kestra.core.storages.kv.KVValueAndMetadata;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Prototype embeddings in Kestra KV",
    description = "Stores embeddings in-memory and persists them to Kestra namespace KV on completion. Suitable for small demos; not scalable. `drop=true` discards any previously saved KV snapshot."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Ingest documents into a KV embedding store.\\nWARNING: the `KestraKVStore` embeddings are for quick prototyping only; since they are stored in a KV store and loaded into memory, this won't scale with large numbers of documents.",
            code = """
                id: document_ingestion
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
                """
        ),
    },
    aliases = "io.kestra.plugin.langchain4j.embeddings.KestraKVStore"
)
public class KestraKVStore extends EmbeddingStoreProvider {
    @JsonIgnore
    private transient InMemoryEmbeddingStore<TextSegment> embeddingStore;

    @Schema(title = "The name of the KV pair to use")
    @Builder.Default
    private Property<String> kvName = Property.ofExpression("{{flow.id}}-embedding-store");

    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) throws IOException, IllegalVariableEvaluationException {
        String key = runContext.render(kvName).as(String.class).orElseThrow();
        Optional<KVEntry> kvEntry = runContext.namespaceKv(runContext.flowInfo().namespace()).get(key);
        if (!drop && kvEntry.isPresent()) {
            try {
                Optional<KVValue> value = runContext.namespaceKv(runContext.flowInfo().namespace()).getValue(kvEntry.get().key());
                embeddingStore = InMemoryEmbeddingStore.fromJson(Objects.requireNonNull(value.orElseThrow().value()).toString());
            } catch (ResourceExpiredException ree) {
                // Should not happen as we didn't set any expiry on the KV
                throw new IOException(ree);
            }
        } else {
            embeddingStore = new InMemoryEmbeddingStore<>();
        }

        return embeddingStore;
    }

    @Override
    public Map<String, Object> outputs(RunContext runContext) throws IOException, IllegalVariableEvaluationException {
        String key = runContext.render(kvName).as(String.class).orElseThrow();
        String storeContent = embeddingStore.serializeToJson();

        KVValueAndMetadata kvValueAndMetadata = new KVValueAndMetadata(null, storeContent);
        runContext.namespaceKv(runContext.flowInfo().namespace()).put(key, kvValueAndMetadata);

        return Map.of("kvName", key);
    }
}
