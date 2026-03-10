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
class WeaviateTest extends ContainerTest {

    private static final String WEAVIATE_OBJECT_CLASS = "embeddings";
    private static final String WEAVIATE_MODEL = "chroma/all-minilm-l6-v2-f32"; // works with Weaviate
    private static final String WEAVIATE_API_KEY = System.getenv("WEAVIATE_API_KEY");

    @Inject
    private RunContextFactory runContextFactory;

    private static GenericContainer<?> weaviate;

    @BeforeAll
    @EnabledIfEnvironmentVariable(named = "WEAVIATE_API_KEY", matches = ".*")
    static void startWeaviate() {
        weaviate = new GenericContainer<>("cr.weaviate.io/semitechnologies/weaviate:1.31.0")
            .withEnv(WEAVIATE_API_KEY != null ? Map.of("WEAVIATE_API_KEY", WEAVIATE_API_KEY) : Map.of())
            .waitingFor(Wait.forListeningPort().withStartupTimeout(Duration.ofSeconds(10))); // to prevent flakiness

        weaviate.start();
    }

    @AfterAll
    @EnabledIfEnvironmentVariable(named = "WEAVIATE_API_KEY", matches = ".*")
    static void stopWeaviate() {
        weaviate.stop();
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "WEAVIATE_API_KEY", matches = ".*")
    void inlineDocuments() throws Exception {
        var runContext = runContextFactory.of(
            Map.of(
                "modelName", WEAVIATE_MODEL,
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
                Weaviate.builder()
                    .apiKey(Property.ofValue(WEAVIATE_API_KEY))
                    .host(Property.ofValue(weaviate.getHost()))
                    .objectClass(Property.ofValue(WEAVIATE_OBJECT_CLASS))
                    .build()
            )
            .fromDocuments(List.of(IngestDocument.InlineDocument.builder().content(Property.ofValue("Everything-as-Code, and from the UI")).build()))
            .build();

        var output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);
    }
}
