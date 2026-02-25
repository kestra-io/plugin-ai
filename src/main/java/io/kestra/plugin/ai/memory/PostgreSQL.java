package io.kestra.plugin.ai.memory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.data.message.ChatMessageDeserializer;
import dev.langchain4j.data.message.ChatMessageSerializer;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.MemoryProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.io.IOException;
import java.sql.*;
import java.time.Duration;
import java.time.Instant;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Persist chat memory in PostgreSQL",
    description = """
        Stores chat history in a PostgreSQL table keyed by `memory_id`; TTL sets an `expires_at` timestamp. Honors drop policies BEFORE/AFTER run; defaults to KEEP. Table is created if missing. Requires reachable Postgres with permissions to create tables."""
)
@Plugin(
    examples = {
        @Example(
            title = "Use PostgreSQL-based chat memory for a conversation",
            full = true,
            code = """
                id: chat_with_memory
                namespace: company.ai

                tasks:
                  - id: first
                    type: io.kestra.plugin.ai.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    memory:
                      type: io.kestra.plugin.ai.memory.PostgreSQL
                      host: localhost
                      port: 5432
                      database: ai_memory
                      user: postgres
                      password: secret
                      tableName: my_custom_memory_table
                    systemMessage: You are a helpful assistant, answer concisely
                    prompt: "{{inputs.first}}"
                """
        )
    }
)
public class PostgreSQL extends MemoryProvider {

    @JsonIgnore
    private transient ChatMemory chatMemory;

    @NotNull
    @Schema(title = "PostgreSQL host", description = "The hostname of your PostgreSQL server")
    private Property<String> host;

    @Schema(title = "PostgreSQL port", description = "The port of your PostgreSQL server")
    @Builder.Default
    private Property<Integer> port = Property.ofValue(5432);

    @NotNull
    @Schema(title = "Database name", description = "The name of the PostgreSQL database")
    private Property<String> database;

    @NotNull
    @Schema(title = "Database user", description = "The username to connect to PostgreSQL")
    private Property<String> user;

    @NotNull
    @Schema(title = "Database password", description = "The password to connect to PostgreSQL")
    private Property<String> password;

    @Schema(
        title = "Table name",
        description = "The name of the table used to store chat memory. Defaults to 'chat_memory'."
    )
    @Builder.Default
    private Property<String> tableName = Property.ofValue("chat_memory");

    @Override
    public ChatMemory chatMemory(RunContext runContext) throws IllegalVariableEvaluationException, IOException {
        var config = resolvedConfig(runContext);

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(
            runContext.render(this.getMessages()).as(Integer.class).orElseThrow()
        );

        var key = runContext.render(this.getMemoryId()).as(String.class).orElseThrow();

        try (Connection conn = connection(config)) {
            ensureTableExists(conn, config.tableName());

            try (PreparedStatement stmt = conn.prepareStatement(
                "SELECT memory_json, expires_at FROM " + config.tableName() + " WHERE memory_id = ?"
            )) {
                stmt.setString(1, key);
                ResultSet rs = stmt.executeQuery();

                if (rs.next()) {
                    Timestamp expiresAt = rs.getTimestamp("expires_at");
                    if (expiresAt == null || expiresAt.toInstant().isAfter(Instant.now())) {
                        if (config.drop() == Drop.BEFORE_TASKRUN) {
                            deleteMemory(conn, key, config.tableName());
                        } else {
                            String json = rs.getString("memory_json");
                            var messages = ChatMessageDeserializer.messagesFromJson(json);
                            messages.forEach(chatMemory::add);
                        }
                    } else {
                        deleteMemory(conn, key, config.tableName());
                    }
                }
            }
        } catch (SQLException e) {
            throw new IOException("Failed to load chat memory from PostgreSQL", e);
        }

        return chatMemory;
    }

    @Override
    public void close(RunContext runContext) throws IllegalVariableEvaluationException, IOException {
        if (chatMemory == null) return;

        var config = resolvedConfig(runContext);
        var key = runContext.render(this.getMemoryId()).as(String.class).orElseThrow();
        var ttl = runContext.render(this.getTtl()).as(Duration.class).orElse(Duration.ofMinutes(10));

        try (Connection conn = connection(config)) {
            ensureTableExists(conn, config.tableName());

            if (config.drop() == Drop.AFTER_TASKRUN) {
                deleteMemory(conn, key, config.tableName());
            } else {
                String memoryJson = ChatMessageSerializer.messagesToJson(chatMemory.messages());
                Instant expiresAt = Instant.now().plus(ttl);

                try (PreparedStatement upsert = conn.prepareStatement(
                    "INSERT INTO " + config.tableName() + " (memory_id, memory_json, expires_at) " +
                        "VALUES (?, ?, ?) " +
                        "ON CONFLICT (memory_id) " +
                        "DO UPDATE SET memory_json = EXCLUDED.memory_json, expires_at = EXCLUDED.expires_at"
                )) {
                    upsert.setString(1, key);
                    upsert.setString(2, memoryJson);
                    upsert.setTimestamp(3, Timestamp.from(expiresAt));
                    upsert.executeUpdate();
                }
            }
        } catch (SQLException e) {
            throw new IOException("Failed to save chat memory to PostgreSQL", e);
        }
    }

    private record ResolvedConfig(String host, int port, String database, String user, String password,
                                  String tableName, Drop drop) {
    }

    private ResolvedConfig resolvedConfig(RunContext runContext) throws IllegalVariableEvaluationException {
        var rHost = runContext.render(this.getHost()).as(String.class).orElseThrow();
        var rPort = runContext.render(this.getPort()).as(Integer.class).orElse(5432);
        var rDb = runContext.render(this.getDatabase()).as(String.class).orElseThrow();
        var rUser = runContext.render(this.getUser()).as(String.class).orElseThrow();
        var rPass = runContext.render(this.getPassword()).as(String.class).orElseThrow();
        var rDrop = runContext.render(this.getDrop()).as(Drop.class).orElse(Drop.NEVER);
        var rTable = runContext.render(this.getTableName()).as(String.class).orElse("chat_memory");

        return new ResolvedConfig(rHost, rPort, rDb, rUser, rPass, rTable, rDrop);
    }

    private Connection connection(ResolvedConfig cfg) throws SQLException {
        return DriverManager.getConnection(
            String.format("jdbc:postgresql://%s:%d/%s", cfg.host(), cfg.port(), cfg.database()),
            cfg.user(),
            cfg.password()
        );
    }

    private void ensureTableExists(Connection conn, String tableName) throws SQLException {
        try (Statement stmt = conn.createStatement()) {
            stmt.execute("""
                    CREATE TABLE IF NOT EXISTS %s (
                        memory_id VARCHAR(255) PRIMARY KEY,
                        memory_json TEXT NOT NULL,
                        expires_at TIMESTAMP NULL
                    )
                """.formatted(tableName));
        }
    }

    private void deleteMemory(Connection conn, String key, String tableName) throws SQLException {
        try (PreparedStatement stmt = conn.prepareStatement(
            "DELETE FROM " + tableName + " WHERE memory_id = ?"
        )) {
            stmt.setString(1, key);
            stmt.executeUpdate();
        }
    }
}
