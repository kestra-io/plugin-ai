package io.kestra.plugin.ai.embeddings;

import java.time.Duration;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.chromadb.ChromaDBContainer;
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
class ChromaTest extends ContainerTest {

    private static final String CHROMA_COLLECTION_NAME = "embeddings";
    private static final String CHROMA_MODEL = "chroma/all-minilm-l6-v2-f32";

    @Inject
    private RunContextFactory runContextFactory;

    private static GenericContainer<?> chroma;

    @BeforeAll
    static void startChroma() {
        chroma = new ChromaDBContainer("chromadb/chroma:0.5.4")
            .waitingFor(Wait.forListeningPort().withStartupTimeout(Duration.ofSeconds(10))); // to prevent flakiness

        chroma.start();
    }

    @AfterAll
    static void stopChroma() {
        chroma.stop();
    }

    @Test
    void inlineDocuments() throws Exception {
        var runContext = runContextFactory.of(
            Map.of(
                "modelName", CHROMA_MODEL,
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
                Chroma.builder()
                    .baseUrl(Property.ofValue("http://" + chroma.getHost() + ':' + chroma.getFirstMappedPort()))
                    .collectionName(Property.ofValue(CHROMA_COLLECTION_NAME))
                    .build()
            )
            .fromDocuments(List.of(IngestDocument.InlineDocument.builder().content(Property.ofValue("Everything-as-Code, and from the UI")).build()))
            .build();

        var output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);
    }
}
