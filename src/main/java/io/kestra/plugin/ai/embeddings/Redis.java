package io.kestra.plugin.ai.embeddings;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.community.store.embedding.redis.RedisEmbeddingStore;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import redis.clients.jedis.DefaultJedisClientConfig;
import redis.clients.jedis.HostAndPort;
import redis.clients.jedis.JedisPooled;

import java.io.IOException;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Store embeddings in Redis",
    description = "Backs an embedding index with Redis (jedis). Uses `indexName` (defaults to `embedding-index`); `drop=true` clears stored vectors. Ensure Redis deployment supports vector search modules for production use."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Ingest documents into a Redis embedding store",
            code = """
                id: document_ingestion
                namespace: company.ai

                tasks:
                  - id: ingest
                    type: io.kestra.plugin.ai.rag.IngestDocument
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-exp-03-07
                      apiKey: my_api_key
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.Redis
                      host: localhost
                      port: 6379
                      indexName: embeddings
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        )
    }
)
public class Redis extends EmbeddingStoreProvider {

    @NotNull
    @Schema(title = "The database server host")
    private Property<String> host;

    @NotNull
    @Schema(title = "The database server port")
    private Property<Integer> port;

    @Schema(title = "The index name")
    @Builder.Default
    private Property<String> indexName = Property.ofValue("embedding-index");

    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) throws IOException, IllegalVariableEvaluationException {

        var rHost = runContext.render(host).as(String.class).orElseThrow();
        var resolvedPort = runContext.render(port).as(Integer.class).orElseThrow();
        var rIndexName = runContext.render(indexName).as(String.class).orElse("embedding-index");

        var jedis = new JedisPooled(
            new HostAndPort(rHost, resolvedPort),
            DefaultJedisClientConfig.builder().build()
        );

        var store = RedisEmbeddingStore.builder()
            .jedisPooled(jedis)
            .indexName(rIndexName)
            .dimension(dimension)
            .build();

        if (drop) {
            store.removeAll();
        }

        return store;
    }
}
