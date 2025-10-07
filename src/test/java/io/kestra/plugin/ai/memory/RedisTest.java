package io.kestra.plugin.ai.memory;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.IdUtils;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;
import io.kestra.plugin.ai.embeddings.KestraKVStore;
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.rag.ChatCompletion;
import jakarta.inject.Inject;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;

import java.time.Duration;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

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

        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "tinydolphin",
            "endpoint", ollamaEndpoint,
            "labels", Map.of("system", Map.of("correlationId", IdUtils.create()))
        ));

        // First interaction — stores memory in Redis
        var firstMessages = List.of(
            ChatMessage.builder()
                .type(ChatMessageType.SYSTEM)
                .content("You are a helpful assistant that remembers user details.")
                .build(),
            ChatMessage.builder()
                .type(ChatMessageType.USER)
                .content("Hello, my name is John")
                .build()
        );

        var rag = ChatCompletion.builder()
            .chatProvider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(KestraKVStore.builder().build())
            .memory(Redis.builder()
                .host(Property.ofValue(redisHost))
                .port(Property.ofValue(redisPort))
                .ttl(Property.ofValue(Duration.ofMinutes(5)))
                .memoryId(Property.ofValue("test-memory"))
                .build())
            .messages(Property.ofValue(firstMessages))
            .chatConfiguration(ChatConfiguration.builder()
                .temperature(Property.ofValue(0.1))
                .seed(Property.ofValue(123456789))
                .build())
            .build();

        var ragOutput = rag.run(runContext);
        assertThat(ragOutput.getTextOutput()).isNotNull();

        // Second interaction — retrieves memory from Redis
        var secondMessages = List.of(
            ChatMessage.builder()
                .type(ChatMessageType.SYSTEM)
                .content("You are a helpful assistant that remembers user details.")
                .build(),
            ChatMessage.builder()
                .type(ChatMessageType.USER)
                .content("What's my name?")
                .build()
        );

        rag = ChatCompletion.builder()
            .chatProvider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(KestraKVStore.builder().build())
            .memory(Redis.builder()
                .host(Property.ofValue(redisHost))
                .port(Property.ofValue(redisPort))
                .memoryId(Property.ofValue("test-memory"))
                .build())
            .messages(Property.ofValue(secondMessages))
            .chatConfiguration(ChatConfiguration.builder()
                .temperature(Property.ofValue(0.1))
                .seed(Property.ofValue(123456789))
                .build())
            .build();

        ragOutput = rag.run(runContext);
        assertThat(ragOutput.getTextOutput()).isNotNull();
        assertThat(ragOutput.getTextOutput().toLowerCase()).contains("john");
    }
}
