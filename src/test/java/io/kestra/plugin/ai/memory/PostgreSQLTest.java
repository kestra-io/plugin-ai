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
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.utility.DockerImageName;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.time.Duration;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class PostgreSQLTest extends ContainerTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    static PostgreSQLContainer<?> postgres;

    @BeforeAll
    static void startPostgres() {
        postgres = new PostgreSQLContainer<>(DockerImageName.parse("postgres:16"))
            .withDatabaseName("ai_memory")
            .withUsername("postgres")
            .withPassword("secret");
        postgres.start();
    }

    @Test
    void testMemoryWithDefaultTable() throws Exception {
        String pgHost = postgres.getHost();
        Integer pgPort = postgres.getMappedPort(5432);
        String database = postgres.getDatabaseName();
        String user = postgres.getUsername();
        String password = postgres.getPassword();

        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "tinydolphin",
            "endpoint", ollamaEndpoint,
            "labels", Map.of("system", Map.of("correlationId", IdUtils.create()))
        ));

        var firstMessages = List.of(
            ChatMessage.builder()
                .type(ChatMessageType.SYSTEM)
                .content("You are a helpful assistant that remembers what the user says.")
                .build(),
            ChatMessage.builder()
                .type(ChatMessageType.USER)
                .content("Hello, my name is Alice")
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
            .memory(PostgreSQL.builder()
                .host(Property.ofValue(pgHost))
                .port(Property.ofValue(pgPort))
                .database(Property.ofValue(database))
                .user(Property.ofValue(user))
                .password(Property.ofValue(password))
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

        var secondMessages = List.of(
            ChatMessage.builder()
                .type(ChatMessageType.SYSTEM)
                .content("You are a helpful assistant that remembers what the user says.")
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
            .memory(PostgreSQL.builder()
                .host(Property.ofValue(pgHost))
                .port(Property.ofValue(pgPort))
                .database(Property.ofValue(database))
                .user(Property.ofValue(user))
                .password(Property.ofValue(password))
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
        assertThat(ragOutput.getTextOutput().toLowerCase()).contains("alice");

        try (Connection conn = DriverManager.getConnection(
            String.format("jdbc:postgresql://%s:%d/%s", pgHost, pgPort, database), user, password);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery("SELECT COUNT(*) FROM chat_memory")) {
            assertThat(rs.next()).isTrue();
            assertThat(rs.getInt(1)).isGreaterThan(0);
        }
    }

    @Test
    void testMemoryWithCustomTable() throws Exception {
        String pgHost = postgres.getHost();
        Integer pgPort = postgres.getMappedPort(5432);
        String database = postgres.getDatabaseName();
        String user = postgres.getUsername();
        String password = postgres.getPassword();
        String customTable = "custom_chat_memory";

        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "tinydolphin",
            "endpoint", ollamaEndpoint,
            "labels", Map.of("system", Map.of("correlationId", IdUtils.create()))
        ));

        var firstMessages = List.of(
            ChatMessage.builder()
                .type(ChatMessageType.SYSTEM)
                .content("You are a helpful assistant that remembers conversations.")
                .build(),
            ChatMessage.builder()
                .type(ChatMessageType.USER)
                .content("Hello, my name is Bob")
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
            .memory(PostgreSQL.builder()
                .host(Property.ofValue(pgHost))
                .port(Property.ofValue(pgPort))
                .database(Property.ofValue(database))
                .user(Property.ofValue(user))
                .password(Property.ofValue(password))
                .tableName(Property.ofValue(customTable))
                .ttl(Property.ofValue(Duration.ofMinutes(5)))
                .memoryId(Property.ofValue("custom-memory"))
                .build())
            .messages(Property.ofValue(firstMessages))
            .chatConfiguration(ChatConfiguration.builder()
                .temperature(Property.ofValue(0.1))
                .seed(Property.ofValue(123456789))
                .build())
            .build();

        var ragOutput = rag.run(runContext);
        assertThat(ragOutput.getTextOutput()).isNotNull();

        try (Connection conn = DriverManager.getConnection(
            String.format("jdbc:postgresql://%s:%d/%s", pgHost, pgPort, database), user, password);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery("SELECT COUNT(*) FROM " + customTable)) {
            assertThat(rs.next()).isTrue();
            assertThat(rs.getInt(1)).isGreaterThan(0);
        }
    }
}
