package io.kestra.plugin.ai.memory;

import java.time.Duration;
import java.util.Map;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.IdUtils;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.embeddings.KestraKVStore;
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.rag.ChatCompletion;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class RedisTest extends ContainerTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    static GenericContainer<?> redis;

    @BeforeAll
    static void startRedis() {
        redis = new GenericContainer<>(DockerImageName.parse("redis:7.2"))
            .withExposedPorts(6379);
        redis.start();
    }

    @Test
    void testMemory() throws Exception {
        String redisHost = redis.getHost();
        Integer redisPort = redis.getMappedPort(6379);

        RunContext runContext = runContextFactory.of(
            "namespace", Map.of(
                "modelName", "tinydolphin",
                "embeddingModelName", "chroma/all-minilm-l6-v2-f32",
                "endpoint", ollamaEndpoint,
                "labels", Map.of("system", Map.of("correlationId", IdUtils.create()))
            )
        );

        // First prompt - store memory in Redis
        var rag = ChatCompletion.builder()
            .chatProvider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddingProvider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ embeddingModelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(KestraKVStore.builder().build())
            .memory(
                Redis.builder()
                    .host(Property.ofValue(redisHost))
                    .port(Property.ofValue(redisPort))
                    .ttl(Property.ofValue(Duration.ofMinutes(5)))
                    .memoryId(Property.ofValue("test-memory"))
                    .build()
            )
            .prompt(Property.ofValue("Hello, my name is John"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .chatConfiguration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var ragOutput = rag.run(runContext);
        assertThat(ragOutput.getTextOutput()).isNotNull();

        // Second prompt - should retrieve memory from Redis
        rag = ChatCompletion.builder()
            .chatProvider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddingProvider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ embeddingModelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(KestraKVStore.builder().build())
            .memory(
                Redis.builder()
                    .host(Property.ofValue(redisHost))
                    .port(Property.ofValue(redisPort))
                    .memoryId(Property.ofValue("test-memory"))
                    .build()
            )
            .prompt(Property.ofValue("What's my name?"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .chatConfiguration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        ragOutput = rag.run(runContext);
        assertThat(ragOutput.getTextOutput()).isNotNull();
        assertThat(ragOutput.getTextOutput().toLowerCase()).contains("john");
    }

    @Test
    void overlappingRunsKeepTheirOwnMemory() throws Exception {
        var memory = Redis.builder()
            .host(Property.ofValue(redis.getHost()))
            .port(Property.ofValue(redis.getMappedPort(6379)))
            .memoryId(Property.ofExpression("{{ conv }}"))
            .build();
        RunContext rcA = runContextFactory.of(Map.of("conv", "conv-a-" + IdUtils.create()));
        RunContext rcB = runContextFactory.of(Map.of("conv", "conv-b-" + IdUtils.create()));

        ChatMemory a = memory.chatMemory(rcA);
        ChatMemory b = memory.chatMemory(rcB);
        a.add(UserMessage.from("I am Alice"));
        b.add(UserMessage.from("I am Bob"));
        memory.close(rcA, a);
        memory.close(rcB, b);

        assertThat(memory.chatMemory(rcA).messages().toString()).contains("Alice").doesNotContain("Bob");
        assertThat(memory.chatMemory(rcB).messages().toString()).contains("Bob").doesNotContain("Alice");
    }
}
