package io.kestra.plugin.ai.embeddings;

import java.time.Duration;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.mongodb.MongoDBAtlasLocalContainer;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.rag.IngestDocument;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class MongoDBAtlasLocalContainerTest extends ContainerTest {

    private static final String DATABASE_NAME = "test_db";
    private static final String COLLECTION_NAME = "embeddings";
    private static final String INDEX_NAME = "embedding_index";
    private static final Integer MONGODB_PORT = 27017;
    private static GenericContainer<?> mongoDBAtlas;

    @Inject
    private RunContextFactory runContextFactory;

    @BeforeAll
    static void startMongoDBAtlasLocal() {
        mongoDBAtlas = new MongoDBAtlasLocalContainer("mongodb/mongodb-atlas-local:7.0.9")
            .withExposedPorts(MONGODB_PORT)
            .waitingFor(Wait.forListeningPort().withStartupTimeout(Duration.ofSeconds(10))); // to prevent flakiness

        mongoDBAtlas.start();
    }

    @AfterAll
    static void stopMongoDBAtlasLocal() {
        if (mongoDBAtlas != null) {
            mongoDBAtlas.stop();
        }
    }

    @Test
    @Disabled("needs to be run against a free tier MongoDB Atlas instance to be tested (with host, username and password)")
    void inlineDocuments() throws Exception {
        var runContext = runContextFactory.of(
            Map.of(
                "modelName", "chroma/all-minilm-l6-v2-f32",
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
                MongoDBAtlas.builder()
                    .scheme(Property.ofValue("mongodb+srv"))
                    .host(Property.ofValue("your_cluster.mongodb.net")) // ends with ".mongodb.net"
                    .username(Property.ofValue("your_username"))
                    .password(Property.ofValue("your_password"))
                    .database(Property.ofValue(DATABASE_NAME))
                    .collectionName(Property.ofValue(COLLECTION_NAME))
                    .indexName(Property.ofValue(INDEX_NAME))
                    .build()
            )
            .fromDocuments(
                List.of(
                    IngestDocument.InlineDocument.builder()
                        .content(Property.ofValue("Test content for embedding."))
                        .build()
                )
            )
            .build();

        var output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);
    }
}
