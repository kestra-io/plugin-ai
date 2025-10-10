package io.kestra.plugin.ai.retriever;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import dev.langchain4j.experimental.rag.content.retriever.sql.SqlDatabaseContentRetriever;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ContentRetrieverProvider;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import javax.sql.DataSource;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(
    title = "SQL Database content retriever using LangChain4j experimental SqlDatabaseContentRetriever. " +
        "⚠ IMPORTANT: the database user should have READ-ONLY permissions."
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
                      modelName: gemini-2.0-flash
                      apiKey: "{{ kv('GOOGLE_API_KEY') }}"
                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.SqlDatabaseRetriever
                        databaseType: POSTGRESQL
                        jdbcUrl: "jdbc:postgresql://localhost:5432/mydb"
                        username: "{{ kv('DB_USER') }}"
                        password: "{{ kv('DB_PASSWORD') }}"
                    prompt: "What are the top 5 customers by revenue?"
                """
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

    @Schema(title = "Type of database to connect to (PostgreSQL, MySQL, or H2)")
    @NotNull
    private Property<DatabaseType> databaseType;

    @Schema(title = "JDBC connection URL to the target database")
    private Property<String> jdbcUrl;

    @Schema(title = "Database username")
    @NotNull
    private Property<String> username;

    @Schema(title = "Database password")
    @NotNull
    private Property<String> password;

    @Schema(title = "Optional JDBC driver class name – automatically resolved if not provided.")
    private Property<String> driver;

    @Schema(title = "Maximum number of database connections in the pool")
    @Builder.Default
    private Property<Integer> maxPoolSize = Property.ofValue(2);

    @Schema(title = "Language model provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @Schema(title = "Language model configuration")
    @NotNull
    @PluginProperty
    @Builder.Default
    private ChatConfiguration configuration = ChatConfiguration.empty();

    @Override
    public ContentRetriever contentRetriever(RunContext runContext) throws IllegalVariableEvaluationException {
        DatabaseType rDatabaseType = runContext.render(this.databaseType).as(DatabaseType.class).orElseThrow();
        String rJdbcUrl = runContext.render(this.jdbcUrl).as(String.class).orElse(null);
        String rUsername = runContext.render(this.username).as(String.class).orElseThrow();
        String rPassword = runContext.render(this.password).as(String.class).orElseThrow();
        String rDriver = runContext.render(this.driver).as(String.class).orElse(null);
        int rMaxPoolSize = runContext.render(this.maxPoolSize).as(Integer.class).orElse(2);

        // Determine default driver if not provided
        if (rDriver == null) {
            rDriver = switch (rDatabaseType) {
                case POSTGRESQL -> "org.postgresql.Driver";
                case MYSQL -> "com.mysql.cj.jdbc.Driver";
                case H2 -> "org.h2.Driver";
            };
        }

        try {
            Class.forName(rDriver);
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("JDBC driver not found on classpath: " + rDriver, e);
        }

        HikariConfig config = new HikariConfig();
        config.setJdbcUrl(rJdbcUrl);
        config.setUsername(rUsername);
        config.setPassword(rPassword);
        config.setDriverClassName(rDriver);
        config.setMaximumPoolSize(rMaxPoolSize);
        config.setPoolName("SqlDatabaseRetrieverPool");

        DataSource dataSource = new HikariDataSource(config);
        ChatModel chatModel = provider.chatModel(runContext, configuration);

        return SqlDatabaseContentRetriever.builder()
            .dataSource(dataSource)
            .chatModel(chatModel)
            .build();
    }
}
