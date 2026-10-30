package io.kestra.plugin.ai.embeddings;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import com.alicloud.openservices.tablestore.model.search.FieldSchema;
import com.alicloud.openservices.tablestore.model.search.FieldType;

import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.VertexAITestEnv;
import io.kestra.plugin.ai.rag.IngestDocument;

import dev.langchain4j.exception.RateLimitException;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.abort;

class TablestoreTest {
    private final String TABLESTORE_ENDPOINT = System.getenv("TABLESTORE_ENDPOINT");
    private final String TABLESTORE_INSTANCE_NAME = System.getenv("TABLESTORE_INSTANCE_NAME");
    private final String TABLESTORE_ACCESS_KEY_ID = System.getenv("TABLESTORE_ACCESS_KEY_ID");
    private final String TABLESTORE_ACCESS_KEY_SECRET = System.getenv("TABLESTORE_ACCESS_KEY_SECRET");
    @Inject
    private RunContextFactory runContextFactory;

    @EnabledIfEnvironmentVariable(named = "TABLESTORE_ENDPOINT", matches = ".*")
    @EnabledIfEnvironmentVariable(named = "TABLESTORE_INSTANCE_NAME", matches = ".*")
    @EnabledIfEnvironmentVariable(named = "TABLESTORE_ACCESS_KEY_ID", matches = ".*")
    @EnabledIfEnvironmentVariable(named = "TABLESTORE_ACCESS_KEY_SECRET", matches = ".*")
    @EnabledIfEnvironmentVariable(named = VertexAITestEnv.CREDENTIALS_ENV, matches = ".+")
    @Test
    void testTablestoreEmbeddingStore_shouldReturnEmbeddingStore() throws Exception {
        RunContext runContext = runContextFactory.of(
            Map.of(
                "endpoint", TABLESTORE_ENDPOINT,
                "instanceName", TABLESTORE_INSTANCE_NAME,
                "accessKeyId", TABLESTORE_ACCESS_KEY_ID,
                "accessKeySecret", TABLESTORE_ACCESS_KEY_SECRET,
                "metadataSchemaList", Arrays.asList(
                    new FieldSchema("meta_example_keyword", FieldType.KEYWORD),
                    new FieldSchema("meta_example_long", FieldType.LONG),
                    new FieldSchema("meta_example_double", FieldType.DOUBLE),
                    new FieldSchema("meta_example_text", FieldType.TEXT)
                        .setAnalyzer(FieldSchema.Analyzer.MaxWord)
                ),
                "flow", Map.of("id", "flow", "namespace", "namespace")
            )
        );

        var task = IngestDocument.builder()
            .provider(
                VertexAITestEnv.embeddingProvider()
            )
            .embeddings(
                Tablestore.builder()
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .instanceName(Property.ofExpression("{{ instanceName }}"))
                    .accessKeyId(Property.ofExpression("{{ accessKeyId }}"))
                    .accessKeySecret(Property.ofExpression("{{ accessKeySecret }}"))
                    .metadataSchemaList(Property.ofExpression("{{ metadataSchemaList }}"))
                    .build()
            )
            .fromDocuments(List.of(IngestDocument.InlineDocument.builder().content(Property.ofValue("I'm Loïc")).build()))
            .build();

        try {
            IngestDocument.Output output = task.run(runContext);
            assertThat(output.getIngestedDocuments()).isEqualTo(1);
        } catch (RateLimitException e) {
            abort("Skipped: Vertex AI rate limited (429)");
        }
    }
}