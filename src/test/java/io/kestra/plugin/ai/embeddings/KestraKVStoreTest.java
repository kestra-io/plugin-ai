package io.kestra.plugin.ai.embeddings;

import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.IdUtils;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class KestraKVStoreTest {
    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void overlappingRunsKeepTheirOwnEmbeddings() throws Exception {
        var embeddings = KestraKVStore.builder().kvName(Property.ofExpression("{{ kv }}")).build();
        String kvA = "kv-a-" + IdUtils.create();
        String kvB = "kv-b-" + IdUtils.create();
        RunContext rcA = runContextFactory.of("io.kestra.plugin.ai.embeddings", Map.of("kv", kvA));
        RunContext rcB = runContextFactory.of("io.kestra.plugin.ai.embeddings", Map.of("kv", kvB));

        var storeA = embeddings.embeddingStore(rcA, 3, false);
        var storeB = embeddings.embeddingStore(rcB, 3, false);
        storeA.add(Embedding.from(new float[] { 1f, 0f, 0f }), TextSegment.from("I am Alice"));
        storeB.add(Embedding.from(new float[] { 0f, 1f, 0f }), TextSegment.from("I am Bob"));
        embeddings.outputs(rcA, storeA);
        embeddings.outputs(rcB, storeB);

        assertThat(storedJson(rcA, kvA)).contains("Alice").doesNotContain("Bob");
        assertThat(storedJson(rcB, kvB)).contains("Bob").doesNotContain("Alice");
    }

    private static String storedJson(RunContext runContext, String key) throws Exception {
        return runContext.namespaceKv(runContext.flowInfo().namespace()).getValue(key).orElseThrow().value().toString();
    }
}
