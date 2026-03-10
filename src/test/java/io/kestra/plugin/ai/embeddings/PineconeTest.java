package io.kestra.plugin.ai.embeddings;

import java.time.Duration;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.rag.IngestDocument;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class PineconeTest extends ContainerTest {

    private static final String PINECONE_INDEX = "embeddings";
    private static final String PINECONE_MODEL = "chroma/all-minilm-l6-v2-f32"; // works with Pinecone
    private static final String PINECONE_API_KEY = System.getenv("PINECONE_API_KEY");
    private static final String PINECONE_CLOUD = "AWS";
    private static final String PINECONE_REGION = "us-east-1";

    @Inject
    private RunContextFactory runContextFactory;

    private static GenericContainer<?> pinecone;

    @BeforeAll
    @EnabledIfEnvironmentVariable(named = "PINECONE_API_KEY", matches = ".*")
    static void startPinecone() {
        pinecone = new GenericContainer<>("ghcr.io/pinecone-io/pinecone-index:latest")
            .withEnv(PINECONE_API_KEY != null ? Map.of("PINECONE_API_KEY", PINECONE_API_KEY) : Map.of())
            .waitingFor(Wait.forListeningPort().withStartupTimeout(Duration.ofSeconds(10))); // to prevent flakiness

        pinecone.start();
    }

    @AfterAll
    @EnabledIfEnvironmentVariable(named = "PINECONE_API_KEY", matches = ".*")
    static void stopPinecone() {
        pinecone.stop();
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "PINECONE_API_KEY", matches = ".*")
    void inlineDocuments() throws Exception {
        var runContext = runContextFactory.of(
            Map.of(
                "modelName", PINECONE_MODEL,
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
                Pinecone.builder()
                    .apiKey(Property.ofValue(PINECONE_API_KEY))
                    .cloud(Property.ofValue(PINECONE_CLOUD))
                    .region(Property.ofValue(PINECONE_REGION))
                    .index(Property.ofValue(PINECONE_INDEX))
                    .build()
            )
            .fromDocuments(List.of(IngestDocument.InlineDocument.builder().content(Property.ofValue("Everything-as-Code, and from the UI")).build()))
            .build();

        var output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);
    }
}
