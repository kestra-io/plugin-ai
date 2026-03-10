package io.kestra.plugin.ai.embeddings;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.milvus.MilvusContainer;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.rag.IngestDocument;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class MilvusTest extends ContainerTest {

    private static final String MILVUS_COLLECTION_NAME = "embeddings";
    private static final String MILVUS_MODEL = "chroma/all-minilm-l6-v2-f32"; // works with Milvus
    private static final String MILVUS_TOKEN = UUID.randomUUID().toString();
    private static final int MILVUS_PORT = 19530;
    private static final int MILVUS_HTTP_PORT = 9091;

    @Inject
    private RunContextFactory runContextFactory;

    private static GenericContainer<?> milvus;

    @BeforeAll
    static void startMilvus() {
        milvus = new MilvusContainer("milvusdb/milvus:v2.3.1")
            .withExposedPorts(MILVUS_PORT, MILVUS_HTTP_PORT)
            .withEnv(Map.of("MILVUS_TOKEN", MILVUS_TOKEN))
            .waitingFor(Wait.forListeningPort().withStartupTimeout(Duration.ofSeconds(10))); // to prevent flakiness

        milvus.start();
    }

    @AfterAll
    static void stopMilvus() {
        milvus.stop();
    }

    @Test
    void inlineDocuments() throws Exception {
        var runContext = runContextFactory.of(
            Map.of(
                "modelName", MILVUS_MODEL,
                "endpoint", ollamaEndpoint,
                "flow", Map.of("id", "flow", "namespace", "namespace")
            )
        );

        var mappedPort = milvus.getMappedPort(MILVUS_PORT);
        var task = IngestDocument.builder()
            .provider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(
                Milvus.builder()
                    .token(Property.ofValue(MILVUS_TOKEN))
                    .uri(Property.ofValue("tcp://" + milvus.getHost() + ':' + mappedPort))
                    .host(Property.ofValue(milvus.getHost()))
                    .port(Property.ofValue(mappedPort))
                    .collectionName(Property.ofValue(MILVUS_COLLECTION_NAME))
                    .build()
            )
            .fromDocuments(List.of(IngestDocument.InlineDocument.builder().content(Property.ofValue("Everything-as-Code, and from the UI")).build()))
            .build();

        var output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);
    }
}
