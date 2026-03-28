package io.kestra.plugin.ai.embeddings;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.testcontainers.containers.MariaDBContainer;
import org.testcontainers.utility.DockerImageName;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.provider.GoogleGemini;
import io.kestra.plugin.ai.rag.IngestDocument;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class MariaDBTest extends ContainerTest {

    @Inject
    private RunContextFactory runContextFactory;
    private static final String GEMINI_API_KEY = System.getenv("GEMINI_API_KEY");
    private static final DockerImageName DEFAULT_IMAGE = DockerImageName.parse("mariadb:11.7-rc");

    private static MariaDBContainer mariaDBContainer;

    @BeforeAll
    static void startMariaDB() {
        mariaDBContainer = new MariaDBContainer<>(DEFAULT_IMAGE).withReuse(true);
        mariaDBContainer.start();
    }

    @AfterAll
    static void stopMariaDB() {
        mariaDBContainer.stop();
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
    void testMariaDbEmbeddingStore_shouldReturnEmbeddingStore() throws Exception {
        String tableName = "embeddings_" + System.nanoTime();
        RunContext runContext = runContextFactory.of(
            Map.of(
                "apiKey", GEMINI_API_KEY,
                "modelName", "gemini-embedding-004",
                "tableName", tableName,
                "flow", Map.of("id", "flow", "namespace", "namespace")
            )
        );

        var task = IngestDocument.builder()
            .provider(
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .build()
            )
            .embeddings(
                MariaDB.builder()
                    .username(Property.ofValue(mariaDBContainer.getUsername()))
                    .password(Property.ofValue(mariaDBContainer.getPassword()))
                    .databaseUrl(Property.ofValue(mariaDBContainer.getJdbcUrl()))
                    .tableName(Property.ofExpression("{{ tableName }}"))
                    .fieldName(Property.ofValue("id"))
                    .createTable(Property.ofValue(true))
                    .build()
            )
            .fromDocuments(List.of(IngestDocument.InlineDocument.builder().content(Property.ofValue("I'm Loïc")).build()))
            .build();

        IngestDocument.Output output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);

    }

    @Test
    @EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
    void testMariaDbEmbeddingStore_givenMetaStoreConfig_shouldReturnEmbeddingStore() throws Exception {
        String tableName = "embeddings_meta_" + System.nanoTime();
        RunContext runContext = runContextFactory.of(
            Map.of(
                "apiKey", GEMINI_API_KEY,
                "modelName", "gemini-embedding-004",
                "tableName", tableName,
                "flow", Map.of("id", "flow", "namespace", "namespace")
            )
        );

        var task = IngestDocument.builder()
            .provider(
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .build()
            )
            .embeddings(
                MariaDB.builder()
                    .username(Property.ofValue(mariaDBContainer.getUsername()))
                    .password(Property.ofValue(mariaDBContainer.getPassword()))
                    .databaseUrl(Property.ofValue(mariaDBContainer.getJdbcUrl()))
                    .tableName(Property.ofExpression("{{ tableName }}"))
                    .fieldName(Property.ofValue("id"))
                    .createTable(Property.ofValue(true))
                    .columnDefinitions(
                        Property.ofValue(
                            Arrays.asList(
                                "source VARCHAR(255)",
                                "timestamp DATETIME",
                                "type VARCHAR(100)"
                            )
                        )
                    ).indexes(Property.ofValue(Arrays.asList("source", "type")))
                    .metadataStorageMode(Property.ofValue("COLUMN_PER_KEY"))
                    .build()
            )
            .fromDocuments(List.of(IngestDocument.InlineDocument.builder().content(Property.ofValue("I'm Loïc")).build()))
            .build();

        IngestDocument.Output output = task.run(runContext);
        assertThat(output.getIngestedDocuments()).isEqualTo(1);
        assertThat(getPrimaryKeyColumn(tableName)).isEqualTo("id");
    }

    private String getPrimaryKeyColumn(String tableName) throws Exception {
        try (
            Connection connection = mariaDBContainer.createConnection("");
            PreparedStatement statement = connection.prepareStatement(
                """
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = ?
                      AND COLUMN_KEY = 'PRI'
                    """
            )
        ) {
            statement.setString(1, tableName);

            try (ResultSet resultSet = statement.executeQuery()) {
                return resultSet.next() ? resultSet.getString("COLUMN_NAME") : null;
            }
        }
    }
}
