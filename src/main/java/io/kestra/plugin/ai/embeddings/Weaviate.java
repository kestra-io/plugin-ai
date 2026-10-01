package io.kestra.plugin.ai.embeddings;

import java.io.IOException;
import java.util.List;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.weaviate.WeaviateEmbeddingStore;
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
    title = "Store embeddings in Weaviate",
    description = "Connects to a Weaviate cluster (HTTP + optional gRPC) using the given host/scheme. `apiKey`, `host`, `port`, and `objectClass` are required. Defaults: scheme \"https\", avoidDups true, consistency QUORUM, secured gRPC true. `drop=true` clears the class contents."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Ingest documents into a Weaviate embedding store",
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
                      type: io.kestra.plugin.ai.embeddings.Weaviate
                      apiKey: "{{ secret('WEAVIATE_API_KEY') }}"
                      scheme: https                                 # http | https (defaults to https)
                      host: your-cluster-id.weaviate.network        # no protocol
                      port: 443                                     # required (e.g. 443 for https, 80 for http)
                      objectClass: Documents                        # required; must start with an uppercase letter
                    drop: true
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        )
    },
    aliases = "io.kestra.plugin.langchain4j.embeddings.Weaviate"
)
public class Weaviate extends EmbeddingStoreProvider {

    @Schema(
        title = "API key",
        description = "Weaviate API key used to authenticate against the cluster. Store it as a Kestra secret rather than inline. No default: this property is required.",
        example = "{{ secret('WEAVIATE_API_KEY') }}"
    )
    @NotNull
    @PluginProperty(secret = true, group = "main")
    private Property<String> apiKey;

    @Schema(
        title = "Scheme",
        description = "Protocol used to reach the cluster: `https` (recommended) or `http`. Defaults to `https`.",
        example = "https"
    )
    @PluginProperty(group = "advanced")
    private Property<String> scheme;

    @Schema(
        title = "Host",
        description = "Cluster hostname, without protocol or port. No default: this property is required.",
        example = "abc123.weaviate.network"
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<String> host;

    @Schema(
        title = "Port",
        description = "Port of the Weaviate HTTP endpoint, typically `443` for `https` and `80` or `8080` for `http`. No default: this property is required.",
        example = "443"
    )
    @NotNull
    @PluginProperty(group = "connection")
    private Property<Integer> port;

    @Schema(
        title = "Object class",
        description = "Weaviate class that stores the embedded objects. It must start with an uppercase letter. No default: this property is required.",
        example = "Documents"
    )
    @NotNull
    @PluginProperty(group = "advanced")
    private Property<String> objectClass;

    @Schema(
        title = "Consistency level",
        description = "Write consistency applied to each object: `ONE`, `QUORUM` or `ALL`. Defaults to `QUORUM`.",
        example = "QUORUM"
    )
    @PluginProperty(group = "advanced")
    private Property<ConsistencyLevel> consistencyLevel;

    @Schema(
        title = "Avoid duplicates",
        description = "If `true`, each object ID is derived from a hash of its text segment, so re-ingesting the same text overwrites the existing object instead of duplicating it. If `false`, a random ID is assigned. Defaults to `true`.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> avoidDups;

    @Schema(
        title = "Metadata field name",
        description = "Property used to store document metadata on the object. Not set by default, in which case the Weaviate client's own default field name applies.",
        example = "_metadata"
    )
    @PluginProperty(group = "advanced")
    private Property<String> metadataFieldName;

    @Schema(
        title = "Metadata keys",
        description = "Metadata keys to persist alongside each object. Defaults to an empty list, meaning no metadata is stored.",
        example = "[\"source\", \"author\"]"
    )
    @PluginProperty(group = "advanced")
    private Property<List<String>> metadataKeys;

    @Schema(
        title = "Use gRPC for batch inserts",
        description = "If `true`, batch inserts go over gRPC, which is faster for large ingestions; searches still use HTTP. Requires `grpcPort`. Defaults to `false`.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> useGrpcForInserts;

    @Schema(
        title = "Secure gRPC",
        description = "Whether the gRPC connection uses TLS. Defaults to `true`. Only relevant when `useGrpcForInserts` is `true`.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> securedGrpc;

    @Schema(
        title = "gRPC port",
        description = "Port of the Weaviate gRPC endpoint. Required when `useGrpcForInserts` is `true`. Not set by default.",
        example = "50051"
    )
    @PluginProperty(group = "connection")
    private Property<Integer> grpcPort;

    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) throws IOException, IllegalVariableEvaluationException {

        // dimension is useless since the given embedding dimension will be used inside Weaviate

        var store = WeaviateEmbeddingStore.builder()
            .apiKey(runContext.render(apiKey).as(String.class).orElseThrow())
            .scheme(runContext.render(scheme).as(String.class).orElse("https"))
            .host(runContext.render(host).as(String.class).orElseThrow())
            .port(runContext.render(port).as(Integer.class).orElseThrow())
            .objectClass(runContext.render(objectClass).as(String.class).orElseThrow())
            .avoidDups(runContext.render(avoidDups).as(Boolean.class).orElse(true))
            .consistencyLevel(runContext.render(consistencyLevel).as(ConsistencyLevel.class).orElse(ConsistencyLevel.QUORUM).name())
            .metadataFieldName(runContext.render(metadataFieldName).as(String.class).orElse(null))
            .metadataKeys(runContext.render(metadataKeys).asList(String.class))
            .useGrpcForInserts(runContext.render(useGrpcForInserts).as(Boolean.class).orElse(false))
            .securedGrpc(runContext.render(securedGrpc).as(Boolean.class).orElse(true))
            .grpcPort(runContext.render(grpcPort).as(Integer.class).orElse(null))
            .build();

        if (drop) {
            store.removeAll();
        }

        return store;
    }

    enum ConsistencyLevel {
        ONE,
        QUORUM,
        ALL,
    }
}
