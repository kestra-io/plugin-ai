package io.kestra.plugin.ai.embeddings;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutionException;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.rag.IngestDocument;

import io.qdrant.client.QdrantClient;
import io.qdrant.client.QdrantGrpcClient;
import io.qdrant.client.grpc.Collections;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class QdrantTest extends ContainerTest {

    private static final String QDRANT_API_KEY = UUID.randomUUID().toString();
    private static final String QDRANT_HOST = "localhost";
    private static final int QDRANT_GRPC_PORT = 6334;
    private static final String QDRANT_COLLECTION_NAME = "embeddings";

    @Inject
    private RunContextFactory runContextFactory;

    private static GenericContainer<?> qdrant;

    @BeforeAll
    static void startQdrant() throws ExecutionException, InterruptedException {
        qdrant = new GenericContainer<>("qdrant/qdrant:v1.14.1")
            .withExposedPorts(QDRANT_GRPC_PORT)
            .withEnv(Map.of("QDRANT__SERVICE__API_KEY", QDRANT_API_KEY))
            .waitingFor(Wait.forListeningPort().withStartupTimeout(Duration.ofSeconds(10))); // to prevent flakiness

        qdrant.start();

        var client = new QdrantClient(
            QdrantGrpcClient.newBuilder(qdrant.getHost(), qdrant.getMappedPort(QDRANT_GRPC_PORT), false)
                .withApiKey(QDRANT_API_KEY)
                .build()
        );

        client.createCollectionAsync(
            QDRANT_COLLECTION_NAME,
            Collections.VectorParams.newBuilder()
                .setDistance(Collections.Distance.Cosine)
                .setSize(2048) // tinydolphin model below produces 2048-dimension embeddings
                .build()
        ).get();

        client.close();
    }

    @AfterAll
    static void stopQdrant() {
        qdrant.stop();
    }

    @Test
    void inlineDocuments() throws Exception {
        var runContext = runContextFactory.of(
            Map.of(
                "modelName", "tinydolphin",
                "endpoint", ollamaEndpoint,
                "flow", Map.of("id", "flow", "namespace", "namespace")
            )
        );

        var task = IngestDocument.builder()
            .provider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(
                Qdrant.builder()
                    .apiKey(Property.ofValue(QDRANT_API_KEY))
                    .host(Property.ofValue(QDRANT_HOST))
                    .port(Property.ofValue(qdrant.getMappedPort(QDRANT_GRPC_PORT)))
                    .collectionName(Property.ofValue(QDRANT_COLLECTION_NAME))
                    .build()
            )
            .fromDocuments(List.of(IngestDocument.InlineDocument.builder().content(Property.ofValue("Everything-as-Code, and from the UI")).build()))
            .drop(Property.ofValue(true))
            .build();

        var output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);
    }
}
