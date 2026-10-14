package io.kestra.plugin.ai.retriever;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.utility.DockerImageName;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.provider.OpenAI;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class SqlDatabaseRetrieverTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void shouldNotLeavePoolThreadWhenContentRetrieverIsBuilt() throws Exception {
        try (PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>(DockerImageName.parse("postgres:16-alpine"))) {
            postgres.start();

            try (
                Connection conn = DriverManager.getConnection(postgres.getJdbcUrl(), postgres.getUsername(), postgres.getPassword());
                Statement stmt = conn.createStatement()
            ) {
                stmt.execute("CREATE TABLE customers (id SERIAL PRIMARY KEY, name VARCHAR(255));");
            }

            RunContext runContext = runContextFactory.of(Map.of());

            var retriever = SqlDatabaseRetriever.builder()
                .databaseType(Property.ofValue(SqlDatabaseRetriever.DatabaseType.POSTGRESQL))
                .jdbcUrl(Property.ofValue(postgres.getJdbcUrl()))
                .username(Property.ofValue(postgres.getUsername()))
                .password(Property.ofValue(postgres.getPassword()))
                .provider(
                    OpenAI.builder()
                        .type(OpenAI.class.getName())
                        .apiKey(Property.ofValue("dummy"))
                        .modelName(Property.ofValue("gpt-4o-mini"))
                        .build()
                )
                .build();

            ContentRetriever contentRetriever = retriever.contentRetriever(runContext);

            assertThat(contentRetriever).isNotNull();
            assertThat(Thread.getAllStackTraces().keySet())
                .extracting(Thread::getName)
                .filteredOn(name -> name.startsWith("SqlDatabaseRetrieverPool"))
                .isEmpty();
        }
    }
}
