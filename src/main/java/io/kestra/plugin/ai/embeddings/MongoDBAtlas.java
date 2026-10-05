package io.kestra.plugin.ai.embeddings;

import java.io.IOException;
import java.time.Duration;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.mongodb.client.MongoClients;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.Await;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.mongodb.IndexMapping;
import dev.langchain4j.store.embedding.mongodb.MongoDbEmbeddingStore;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import io.kestra.core.models.annotations.PluginProperty;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Store embeddings in MongoDB Atlas",
    description = "Uses MongoDB Atlas vector search with the provided collection/index; can optionally create the index and wait for readiness (up to 1 minute). Supply scheme/host and credentials; `drop=true` removes stored vectors."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Ingest documents into a MongoDB Atlas embedding store",
            code = """
                id: document_ingestion
                namespace: company.ai

                tasks:
                  - id: ingest
                    type: io.kestra.plugin.ai.rag.IngestDocument
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-001
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.MongoDBAtlas
                      scheme: mongodb+srv
                      username: "{{ secret('MONGODB_ATLAS_USERNAME') }}"
                      password: "{{ secret('MONGODB_ATLAS_PASSWORD') }}"
                      host: "{{ secret('MONGODB_ATLAS_HOST') }}"
                      database: "{{ secret('MONGODB_ATLAS_DATABASE') }}"
                      collectionName: embeddings
                      indexName: embeddings
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        ),
    },
    aliases = "io.kestra.plugin.langchain4j.embeddings.MongoDBAtlas"
)
public class MongoDBAtlas extends EmbeddingStoreProvider {

    @Schema(
        title = "Username",
        description = "User connecting to the MongoDB Atlas cluster. Not set by default; omit it along with `password` for an unauthenticated connection.",
        example = "atlas_user"
    )
    @PluginProperty(group = "connection")
    private Property<String> username;

    @Schema(
        title = "Password",
        description = "Password of the database user. Store it as a Kestra secret rather than inline. Not set by default; omit it along with `username` for an unauthenticated connection.",
        example = "{{ secret('MONGODB_PASSWORD') }}"
    )
    @PluginProperty(secret = true, group = "connection")
    private Property<String> password;

    @NotNull
    @Schema(
        title = "Connection scheme",
        description = "Scheme of the MongoDB connection string: `mongodb+srv` for Atlas clusters, `mongodb` for a direct connection. No default: this property is required.",
        example = "mongodb+srv"
    )
    @PluginProperty(group = "main")
    private Property<String> scheme;

    @NotNull
    @Schema(
        title = "Host",
        description = "Hostname of the MongoDB cluster, optionally with a port for the `mongodb` scheme. No default: this property is required.",
        example = "cluster0.abcde.mongodb.net"
    )
    @PluginProperty(group = "main")
    private Property<String> host;

    @NotNull
    @Schema(
        title = "Database name",
        description = "Name of the database holding the embeddings collection. No default: this property is required.",
        example = "vectordb"
    )
    @PluginProperty(group = "connection")
    private Property<String> database;

    @Schema(
        title = "Connection string options",
        description = "Extra options appended to the MongoDB connection string as query parameters. Not set by default.",
        example = "{retryWrites: \"true\", w: \"majority\"}"
    )
    @PluginProperty(group = "advanced")
    private Property<Map<String, Object>> options;

    @NotNull
    @Schema(
        title = "Collection name",
        description = "Collection that stores the embedding documents. No default: this property is required.",
        example = "embeddings"
    )
    @PluginProperty(group = "main")
    private Property<String> collectionName;

    @NotNull
    @Schema(
        title = "Index name",
        description = "Name of the Atlas Vector Search index used to query the collection. No default: this property is required.",
        example = "vector_index"
    )
    @PluginProperty(group = "main")
    private Property<String> indexName;

    @Schema(
        title = "Metadata field names",
        description = "Metadata keys to map into the vector search index so they can be filtered on. Not set by default, in which case no metadata index mapping is created.",
        example = "[\"source\", \"author\"]"
    )
    @PluginProperty(group = "advanced")
    private Property<List<String>> metadataFieldNames;

    @Schema(
        title = "Create the index",
        description = "If `true`, create the Atlas Vector Search index when it does not already exist. Defaults to `false`, which assumes the index is provisioned beforehand.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> createIndex;

    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) throws IOException, IllegalVariableEvaluationException {

        var mongoClient = MongoClients.create(buildUri(runContext));

        var renderedCreateIndex = runContext.render(createIndex).as(Boolean.class).orElse(false);
        var store = MongoDbEmbeddingStore.builder()
            .fromClient(mongoClient)
            .databaseName(runContext.render(database).as(String.class).orElseThrow())
            .collectionName(runContext.render(collectionName).as(String.class).orElseThrow())
            .indexName(runContext.render(indexName).as(String.class).orElseThrow())
            .createIndex(renderedCreateIndex)
            .indexMapping(
                metadataFieldNames != null ? IndexMapping.builder()
                    .dimension(dimension)
                    .metadataFieldNames(new HashSet<>(runContext.render(metadataFieldNames).asList(String.class)))
                    .build() : null
            )
            .build();

        if (renderedCreateIndex) {
            // Creating a vector search index can take up to a minute, so this delay allows the index to become queryable
            try {
                Await.until(
                    () ->
                    {
                        try {
                            // Try a harmless dummy query to check index readiness
                            store.search(
                                EmbeddingSearchRequest.builder()
                                    .queryEmbedding(Embedding.from(Collections.nCopies(dimension, 0.0f)))
                                    .maxResults(1)
                                    .build()
                            );
                            return true;
                        } catch (Exception e) {
                            return false;
                        }
                    },
                    Duration.ofSeconds(1),
                    Duration.ofMinutes(1)
                );
            } catch (TimeoutException | RuntimeException e) {
                throw new RuntimeException("MongoDB vector index was not ready within 1 minute.", e);
            }
        }

        if (drop) {
            store.removeAll();
        }

        return store;
    }

    private String buildUri(RunContext runContext) throws IllegalVariableEvaluationException {

        // Format: mongodb+srv://[username:password@]host[/[database][?options]]

        var scheme = runContext.render(this.scheme).as(String.class).orElseThrow();
        var username = runContext.render(this.username).as(String.class).orElse(null);
        var password = runContext.render(this.password).as(String.class).orElse(null);
        var host = runContext.render(this.host).as(String.class).orElseThrow();
        var database = runContext.render(this.database).as(String.class).orElseThrow();
        var options = runContext.render(this.options).asMap(String.class, Object.class);

        return scheme + "://" +
            (username != null && password != null ? username + ":" + password + "@" : "") +
            host + "/" +
            database + // optional in the connection string but still required in MongoDbEmbeddingStore code :shrug:
            toMongoOptionsQueryString(options);
    }

    private String toMongoOptionsQueryString(Map<String, Object> options) {
        if (options == null || options.isEmpty()) {
            return "";
        }

        return options.entrySet().stream()
            .map(e -> e.getKey() + "=" + e.getValue())
            .collect(Collectors.joining("&", "?", ""));
    }
}
