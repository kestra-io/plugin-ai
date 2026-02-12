package io.kestra.plugin.ai.memory;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.retrys.Exponential;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.IdUtils;
import io.kestra.core.utils.RetryUtils;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.embeddings.KestraKVStore;
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.rag.ChatCompletion;
import jakarta.inject.Inject;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.utility.DockerImageName;

import java.sql.*;
import java.time.Duration;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
@Execution(ExecutionMode.SAME_THREAD)
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

        // Avoid cross-test / parallel interference
        String memoryId = "test-memory-" + IdUtils.create();

        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "tinydolphin",
            "endpoint", ollamaEndpoint,
            "labels", Map.of("system", Map.of("correlationId", IdUtils.create()))
        ));

        // 1) First prompt: store chat memory
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
                .memoryId(Property.ofValue(memoryId))
                .build())
            .prompt(Property.ofValue("Hello, my name is Alice"))
            .chatConfiguration(ChatConfiguration.builder()
                .temperature(Property.ofValue(0.0))
                .seed(Property.ofValue(123456789))
                .build())
            .build();

        var ragOutput = rag.run(runContext);
        assertThat(ragOutput.getTextOutput()).isNotNull();

        // 2) Wait until close() persisted memory_json for THIS memoryId (RetryUtils)
        RetryUtils.Instance<String, RuntimeException> persistRetry =
            RetryUtils.Instance.<String, RuntimeException>builder()
                .policy(Exponential.builder()
                    .interval(Duration.ofMillis(100))
                    .maxInterval(Duration.ofMillis(200))
                    .delayFactor(2.0)
                    .maxAttempts(25)
                    .build())
                .failureFunction(failed ->
                    new RuntimeException("Memory not persisted for memory_id=" + memoryId, failed))
                .build();

        String memoryJson = persistRetry.run(
            result -> result == null || result.isBlank(),
            () -> {
                try (Connection conn = DriverManager.getConnection(
                    String.format("jdbc:postgresql://%s:%d/%s", pgHost, pgPort, database), user, password);
                     PreparedStatement ps = conn.prepareStatement(
                         "SELECT memory_json FROM chat_memory WHERE memory_id = ?"
                     )) {
                    ps.setString(1, memoryId);
                    try (ResultSet rs = ps.executeQuery()) {
                        if (!rs.next()) return null;
                        return rs.getString("memory_json");
                    }
                }
            }
        );

        assertThat(memoryJson).isNotBlank();
        // Deterministic assertion: persisted memory contains what we wrote
        assertThat(memoryJson.toLowerCase()).contains("alice");

        // (Optional) 3) Second prompt: just ensure reading memory doesn't crash (no assertion on LLM output)
        var rag2 = ChatCompletion.builder()
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
                .memoryId(Property.ofValue(memoryId))
                .build())
            .prompt(Property.ofValue("What's my name?"))
            .chatConfiguration(ChatConfiguration.builder()
                .temperature(Property.ofValue(0.0))
                .seed(Property.ofValue(123456789))
                .build())
            .build();

        var out2 = rag2.run(runContext);
        assertThat(out2.getTextOutput()).isNotNull();

        // 4) Validate data persisted in default table for this memoryId
        try (Connection conn = DriverManager.getConnection(
            String.format("jdbc:postgresql://%s:%d/%s", pgHost, pgPort, database), user, password);
             PreparedStatement ps = conn.prepareStatement(
                 "SELECT COUNT(*) FROM chat_memory WHERE memory_id = ?"
             )) {
            ps.setString(1, memoryId);
            try (ResultSet rs = ps.executeQuery()) {
                assertThat(rs.next()).isTrue();
                assertThat(rs.getInt(1)).isEqualTo(1);
            }
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

        // First prompt : store chat memory in a custom table
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
            .prompt(Property.ofValue("Hello, my name is Bob"))
            .chatConfiguration(ChatConfiguration.builder()
                .temperature(Property.ofValue(0.1))
                .seed(Property.ofValue(123456789))
                .build())
            .build();

        var ragOutput = rag.run(runContext);
        assertThat(ragOutput.getTextOutput()).isNotNull();

        // Verify that the table was created and data stored correctly
        try (Connection conn = DriverManager.getConnection(
            String.format("jdbc:postgresql://%s:%d/%s", pgHost, pgPort, database), user, password);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery("SELECT COUNT(*) FROM " + customTable)) {
            assertThat(rs.next()).isTrue();
            assertThat(rs.getInt(1)).isGreaterThan(0);
        }
    }
}
