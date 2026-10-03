package io.kestra.plugin.ai.provider;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.property.URIFetcher;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.OnnxEmbeddingModel;
import dev.langchain4j.model.embedding.onnx.PoolingMode;
import dev.langchain4j.model.image.ImageModel;
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
    title = "Compute embeddings in-process with an ONNX model",
    description = """
        Runs a BERT-style sentence-embedding model exported to ONNX (for example all-MiniLM, BGE or E5) inside the Kestra worker with ONNX Runtime on CPU: no API key and no model server. \
        Provide the `.onnx` model file and its Hugging Face `tokenizer.json`; `modelName` is only used as a label. \
        Embeddings only: chat completion and image generation are not supported. \
        A loaded model is kept in the worker's memory and reused by every task that provides the same model and tokenizer files."""
)
@Plugin(
    examples = {
        @Example(
            title = "Ingest and search documents with all-MiniLM-L6-v2 downloaded from Hugging Face",
            full = true,
            code = {
                """
                    id: onnx_rag
                    namespace: company.ai

                    tasks:
                      - id: model
                        type: io.kestra.plugin.core.http.Download
                        uri: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx

                      - id: tokenizer
                        type: io.kestra.plugin.core.http.Download
                        uri: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json

                      - id: ingest
                        type: io.kestra.plugin.ai.rag.IngestDocument
                        provider:
                          type: io.kestra.plugin.ai.provider.Onnx
                          modelName: all-MiniLM-L6-v2
                          modelUri: "{{ outputs.model.uri }}"
                          tokenizerUri: "{{ outputs.tokenizer.uri }}"
                        embeddings:
                          type: io.kestra.plugin.ai.embeddings.KestraKVStore
                        drop: true
                        fromDocuments:
                          - content: Kestra is an open-source orchestration platform for data and AI workflows.
                          - content: PostgreSQL is a relational database.

                      - id: search
                        type: io.kestra.plugin.ai.rag.Search
                        provider:
                          type: io.kestra.plugin.ai.provider.Onnx
                          modelName: all-MiniLM-L6-v2
                          modelUri: "{{ outputs.model.uri }}"
                          tokenizerUri: "{{ outputs.tokenizer.uri }}"
                        embeddings:
                          type: io.kestra.plugin.ai.embeddings.KestraKVStore
                        query: Which tool orchestrates workflows?
                        maxResults: 1
                        minScore: 0.3
                        fetchType: FETCH
                    """
            }
        ),
        @Example(
            title = "Use a BGE model stored as namespace files",
            full = true,
            code = {
                """
                    id: onnx_bge_search
                    namespace: company.ai

                    inputs:
                      - id: query
                        type: STRING

                    tasks:
                      - id: search
                        type: io.kestra.plugin.ai.rag.Search
                        provider:
                          type: io.kestra.plugin.ai.provider.Onnx
                          modelName: bge-small-en-v1.5
                          modelUri: nsfile:///models/bge-small-en-v1.5/model.onnx
                          tokenizerUri: nsfile:///models/bge-small-en-v1.5/tokenizer.json
                          poolingMode: CLS
                        embeddings:
                          type: io.kestra.plugin.ai.embeddings.KestraKVStore
                        query: "{{ inputs.query }}"
                        maxResults: 3
                        minScore: 0.5
                        fetchType: FETCH
                    """
            }
        )
    }
)
public class Onnx extends ModelProvider {
    // ONNX Runtime sessions hold native memory that is never released, so each distinct model is loaded once per worker.
    private static final Map<String, EmbeddingModel> MODELS = new ConcurrentHashMap<>();

    @Schema(
        title = "ONNX model file",
        description = "URI of the `.onnx` model file: a namespace file (`nsfile:///path`), an internal storage file (`kestra:///...`, e.g. the output of `io.kestra.plugin.core.http.Download`), or a local file (`file:///path`, must be allowed by `kestra.local-files.allowed-paths`).",
        example = "nsfile:///models/all-MiniLM-L6-v2/model.onnx"
    )
    @NotNull
    @PluginProperty(internalStorageURI = true, group = "main")
    private Property<String> modelUri;

    @Schema(
        title = "Tokenizer file",
        description = "URI of the Hugging Face `tokenizer.json` matching the model, using the same schemes as `modelUri`.",
        example = "nsfile:///models/all-MiniLM-L6-v2/tokenizer.json"
    )
    @NotNull
    @PluginProperty(internalStorageURI = true, group = "main")
    private Property<String> tokenizerUri;

    @Schema(
        title = "Pooling mode",
        description = "How token vectors are combined into one embedding. `MEAN` suits sentence-transformers models such as all-MiniLM and E5; BGE models expect `CLS`. A mismatch does not fail but degrades search quality. Defaults to `MEAN`."
    )
    @Builder.Default
    @PluginProperty(group = "advanced")
    private Property<PoolingMode> poolingMode = Property.ofValue(PoolingMode.MEAN);

    // Skipping Chat and Image Models for ONNX Provider, since we are only configuring it for Embeddings.
    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) {
        throw new UnsupportedOperationException("Onnx only supports embedding models.");
    }

    @Override
    public ImageModel imageModel(RunContext runContext) {
        throw new UnsupportedOperationException("Onnx only supports embedding models.");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        var rPoolingMode = runContext.render(poolingMode).as(PoolingMode.class).orElse(PoolingMode.MEAN);
        var modelFile = copyToWorkingDir(runContext, modelUri, ".onnx");
        var tokenizerFile = copyToWorkingDir(runContext, tokenizerUri, ".json");

        return MODELS.computeIfAbsent(
            modelFile.sha256() + ":" + tokenizerFile.sha256() + ":" + rPoolingMode,
            cacheKey ->
            {
                runContext.logger().info("Loading ONNX embedding model (sha256 {})", modelFile.sha256());
                return new OnnxEmbeddingModel(modelFile.path(), tokenizerFile.path(), rPoolingMode);
            }
        );
    }

    private static LocalFile copyToWorkingDir(RunContext runContext, Property<String> fileUri, String extension) throws IllegalVariableEvaluationException {
        var rFileUri = runContext.render(fileUri).as(String.class).orElseThrow();
        try {
            var sha256Digest = MessageDigest.getInstance("SHA-256");
            var localPath = runContext.workingDir().createTempFile(extension);
            try (var source = new DigestInputStream(URIFetcher.of(rFileUri).fetch(runContext), sha256Digest)) {
                Files.copy(source, localPath, StandardCopyOption.REPLACE_EXISTING);
            }
            return new LocalFile(localPath, HexFormat.of().formatHex(sha256Digest.digest()));
        } catch (IOException | NoSuchAlgorithmException e) {
            throw new IllegalArgumentException("Unable to read ONNX file from uri `" + rFileUri + "`.", e);
        }
    }

    private record LocalFile(Path path, String sha256) {
    }
}
