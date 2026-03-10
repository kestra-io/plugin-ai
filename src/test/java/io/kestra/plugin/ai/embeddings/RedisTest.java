package io.kestra.plugin.ai.embeddings;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;

import com.redis.testcontainers.RedisStackContainer;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.rag.IngestDocument;

import jakarta.inject.Inject;

import static com.redis.testcontainers.RedisStackContainer.DEFAULT_IMAGE_NAME;
import static com.redis.testcontainers.RedisStackContainer.DEFAULT_TAG;
import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
public class RedisTest extends ContainerTest {

    private static final String REDIS_COLLECTION_NAME = "embeddings";
    private static final String REDIS_MODEL = "chroma/all-minilm-l6-v2-f32";

    @Inject
    private RunContextFactory runContextFactory;

    private static GenericContainer<?> redis;

    @BeforeAll
    static void startRedis() {
        redis = new RedisStackContainer(DEFAULT_IMAGE_NAME.withTag(DEFAULT_TAG));
        redis.start();
    }

    @AfterAll
    static void stopRedis() {
        redis.stop();
    }

    @Test
    void inlineDocuments() throws Exception {
        var runContext = runContextFactory.of(
            Map.of(
                "modelName", REDIS_MODEL,
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
                Redis.builder()
                    .host(Property.ofValue(redis.getHost()))
                    .port(Property.ofValue(redis.getFirstMappedPort()))
                    .indexName(Property.ofValue(REDIS_COLLECTION_NAME))
                    .build()
            )
            .fromDocuments(List.of(IngestDocument.InlineDocument.builder().content(Property.ofValue("Everything-as-Code, and from the UI")).build()))
            .build();

        var output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);
    }
}
