package io.kestra.plugin.ai.retriever;

import java.io.PrintWriter;
import java.sql.Connection;
import java.sql.Driver;
import java.sql.SQLException;
import java.sql.SQLFeatureNotSupportedException;
import java.util.Properties;
import java.util.logging.Logger;
import javax.sql.DataSource;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Metric;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.TokenBudgetChatModel;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ContentRetrieverProvider;
import io.kestra.plugin.ai.domain.ModelProvider;

import dev.langchain4j.experimental.rag.content.retriever.sql.SqlDatabaseContentRetriever;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(
    title = "Retrieve context from SQL (experimental)",
    description = """
        Uses LangChain4j’s experimental SqlDatabaseContentRetriever to translate questions into SQL and return rows as context. Requires a read-only JDBC user; supports PostgreSQL/MySQL/H2 with auto driver selection. A new connection is opened for each query and closed right after it."""
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "RAG chat with a SQL Database content retriever (answers grounded in database data)",
            code = """
                id: rag
                namespace: company.ai

                tasks:
                  - id: chat_with_rag_and_sql_retriever
                    type: io.kestra.plugin.ai.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-3.5-flash-lite
                      apiKey: "{{ secret('GOOGLE_API_KEY') }}"
                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.SqlDatabaseRetriever
                        databaseType: POSTGRESQL
                        jdbcUrl: "jdbc:postgresql://localhost:5432/mydb"
                        username: "{{ secret('DB_USER') }}"
                        password: "{{ secret('DB_PASSWORD') }}"
                    prompt: "What are the top 5 customers by revenue?"
                """
        )
    },
    metrics = {
        @Metric(
            name = "ai.provider.calls",
            type = Counter.TYPE,
            unit = "calls",
            description = "Number of times a chat model is obtained from a provider, tagged by provider class name"
        )
    }
)
public class SqlDatabaseRetriever extends ContentRetrieverProvider {

    @Schema(
        title = "Supported database types",
        description = "Determines the default JDBC driver and connection format."
    )
    public enum DatabaseType {
        POSTGRESQL,
        MYSQL,
        H2
    }

    @Schema(
        title = "Database type",
        description = "Database engine to connect to: `POSTGRESQL`, `MYSQL` or `H2`. It selects the default JDBC driver when `driver` is not set. No default: this property is required.",
        example = "POSTGRESQL"
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<DatabaseType> databaseType;

    @Schema(
        title = "JDBC URL",
        description = "JDBC connection URL of the database the model queries. It must be set for the retriever to connect, even though it is not enforced by validation. Each query opens its own connection, so the connect timeout is the driver's: PostgreSQL gives up after 10 seconds by default, while other drivers may wait for the OS TCP timeout unless the URL sets one (for example `connectTimeout=30000` for MySQL, in milliseconds). No default.",
        example = "jdbc:postgresql://localhost:5432/mydb"
    )
    @PluginProperty(group = "connection")
    private Property<String> jdbcUrl;

    @Schema(
        title = "Database username",
        description = "User connecting to the database. Grant it read-only access, since the model generates the SQL that is executed. No default: this property is required.",
        example = "postgres"
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<String> username;

    @Schema(
        title = "Database password",
        description = "Password of the database user. Store it as a Kestra secret rather than inline. No default: this property is required.",
        example = "{{ secret('POSTGRES_PASSWORD') }}"
    )
    @NotNull
    @PluginProperty(secret = true, group = "main")
    private Property<String> password;

    @Schema(
        title = "JDBC driver class name",
        description = "Fully qualified JDBC driver class, which must be on the classpath. Not set by default, in which case it is derived from `databaseType`: `org.postgresql.Driver`, `com.mysql.cj.jdbc.Driver` or `org.h2.Driver`.",
        example = "org.postgresql.Driver"
    )
    @PluginProperty(group = "advanced")
    private Property<String> driver;

    @Schema(
        title = "Maximum connection pool size (deprecated)",
        description = "Deprecated and ignored: no pool is used, a connection is opened for each query and closed right after it.",
        example = "2",
        deprecated = true
    )
    @PluginProperty(group = "execution")
    @Deprecated
    private Property<Integer> maxPoolSize;

    @Schema(
        title = "Language model provider",
        description = "Model provider used to translate the natural-language question into SQL. No default: this property is required.",
        example = "{type: \"io.kestra.plugin.ai.provider.GoogleGemini\", apiKey: \"{{ secret('GEMINI_API_KEY') }}\", modelName: \"gemini-3.5-flash-lite\"}"
    )
    @NotNull
    @PluginProperty(group = "main")
    private ModelProvider provider;

    @Schema(
        title = "Language model configuration",
        description = "Chat model settings (temperature, response format, token limits, and so on) applied to the SQL-generating model. Defaults to an empty configuration, so the provider's own defaults apply.",
        example = "{temperature: 0.1}"
    )
    @NotNull
    @PluginProperty(group = "main")
    @Builder.Default
    private ChatConfiguration configuration = ChatConfiguration.empty();

    @Override
    public ContentRetriever contentRetriever(RunContext runContext) throws IllegalVariableEvaluationException {
        DatabaseType rDatabaseType = runContext.render(this.databaseType).as(DatabaseType.class).orElseThrow();
        String rJdbcUrl = runContext.render(this.jdbcUrl).as(String.class).orElse(null);
        String rUsername = runContext.render(this.username).as(String.class).orElseThrow();
        String rPassword = runContext.render(this.password).as(String.class).orElseThrow();
        String rDriver = runContext.render(this.driver).as(String.class).orElse(null);

        // Determine default driver if not provided
        if (rDriver == null) {
            rDriver = switch (rDatabaseType) {
                case POSTGRESQL -> "org.postgresql.Driver";
                case MYSQL -> "com.mysql.cj.jdbc.Driver";
                case H2 -> "org.h2.Driver";
            };
        }

        Driver jdbcDriver;
        try {
            jdbcDriver = (Driver) Class.forName(rDriver).getDeclaredConstructor().newInstance();
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("JDBC driver not found on classpath: " + rDriver, e);
        } catch (ReflectiveOperationException e) {
            throw new IllegalArgumentException("Cannot instantiate the JDBC driver '%s'.".formatted(rDriver), e);
        }

        if (rJdbcUrl == null) {
            throw new IllegalArgumentException("`jdbcUrl` is required.");
        }

        Properties connectionProperties = new Properties();
        connectionProperties.setProperty("user", rUsername);
        connectionProperties.setProperty("password", rPassword);

        DataSource dataSource = new DriverDataSource(jdbcDriver, rJdbcUrl, connectionProperties);
        ChatModel chatModel = TokenBudgetChatModel.wrap(
            provider.chatModel(runContext, configuration),
            runContext,
            configuration
        );
        runContext.metric(Counter.of("ai.provider.calls", 1, "provider", provider.getClass().getName()));

        return SqlDatabaseContentRetriever.builder()
            .dataSource(dataSource)
            .chatModel(chatModel)
            .build();
    }

    // Not pooled on purpose: LangChain4j closes every connection it gets, and a pool would outlive the run.
    private static final class DriverDataSource implements DataSource {
        private final Driver driver;
        private final String url;
        private final Properties properties;

        private DriverDataSource(Driver driver, String url, Properties properties) {
            this.driver = driver;
            this.url = url;
            this.properties = properties;
        }

        @Override
        public Connection getConnection() throws SQLException {
            Connection connection = driver.connect(url, properties);
            if (connection == null) {
                throw new SQLException("The JDBC driver '%s' does not accept the configured JDBC URL.".formatted(driver.getClass().getName()));
            }
            return connection;
        }

        @Override
        public Connection getConnection(String username, String password) throws SQLException {
            throw new SQLFeatureNotSupportedException();
        }

        @Override
        public PrintWriter getLogWriter() throws SQLException {
            throw new SQLFeatureNotSupportedException();
        }

        @Override
        public void setLogWriter(PrintWriter out) throws SQLException {
            throw new SQLFeatureNotSupportedException();
        }

        @Override
        public void setLoginTimeout(int seconds) throws SQLException {
            throw new SQLFeatureNotSupportedException();
        }

        @Override
        public int getLoginTimeout() {
            return 0;
        }

        @Override
        public Logger getParentLogger() throws SQLFeatureNotSupportedException {
            return driver.getParentLogger();
        }

        @Override
        public <T> T unwrap(Class<T> iface) throws SQLException {
            throw new SQLFeatureNotSupportedException();
        }

        @Override
        public boolean isWrapperFor(Class<?> iface) {
            return false;
        }
    }
}
