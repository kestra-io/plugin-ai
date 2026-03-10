package io.kestra.plugin.ai.embeddings;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.rag.IngestDocument;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class PGVectorTest extends ContainerTest {
    @Inject
    private RunContextFactory runContextFactory;

    private static GenericContainer<?> pg;

    @BeforeAll
    static void startPGVector() {
        pg = new GenericContainer<>("pgvector/pgvector:pg17")
            .withExposedPorts(5432)
            .withEnv(Map.of("POSTGRES_USER", "kestra", "POSTGRES_PASSWORD", "kestra"));

        pg.start();
    }

    @AfterAll
    static void stopPGVector() {
        pg.stop();
    }

    @Test
    void inlineDocuments() throws Exception {
        RunContext runContext = runContextFactory.of(
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
                PGVector.builder()
                    .host(Property.ofValue("localhost"))
                    .port(Property.ofValue(pg.getMappedPort(5432)))
                    .user(Property.ofValue("kestra"))
                    .password(Property.ofValue("kestra"))
                    .database(Property.ofValue("postgres"))
                    .table(Property.ofValue("embeddings"))
                    .build()
            )
            .fromDocuments(List.of(IngestDocument.InlineDocument.builder().content(Property.ofValue("I'm Loïc")).build()))
            .build();

        IngestDocument.Output output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);
    }

}