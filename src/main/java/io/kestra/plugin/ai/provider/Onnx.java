package io.kestra.plugin.ai.provider;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import java.util.function.Supplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.DimensionAwareEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.AbstractInProcessEmbeddingModel;
import dev.langchain4j.model.embedding.onnx.OnnxBertBiEncoder;
import dev.langchain4j.model.embedding.onnx.PoolingMode;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.output.Response;
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
        A loaded model stays in the worker's memory and is reused by every task that provides the same model and tokenizer files. \
        Each worker keeps at most 2 models loaded between embedding calls (each distinct model, tokenizer and `poolingMode` combination counts as one): loading another one unloads the least recently used model as soon as no running call uses it, and that model is loaded again on its next use. \
        Administrators can change the limit with the `max-loaded-models` plugin configuration of `io.kestra.plugin.ai.provider.Onnx`; set it to at least the number of distinct models used at the same time, otherwise they keep unloading and reloading each other."""
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
    private static final int DEFAULT_MAX_LOADED_MODELS = 2;

    // ONNX Runtime sessions hold native memory that only an explicit close() releases, so the plugin owns them and bounds how many stay loaded.
    static final ModelCache MODELS = new ModelCache();

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
        var maxLoadedModels = maxLoadedModels(runContext);
        var modelFile = copyToWorkingDir(runContext, modelUri, ".onnx");
        var tokenizerFile = copyToWorkingDir(runContext, tokenizerUri, ".json");

        return new CachedEmbeddingModel(
            modelFile.sha256() + ":" + tokenizerFile.sha256() + ":" + rPoolingMode,
            maxLoadedModels,
            () ->
            {
                runContext.logger().info("Loading ONNX embedding model (sha256 {})", modelFile.sha256());
                return LoadedModel.load(modelFile.path(), tokenizerFile.path(), rPoolingMode);
            }
        );
    }

    private int maxLoadedModels(RunContext runContext) {
        Optional<Object> configured = runContext.cloneForPlugin(this).pluginConfiguration("max-loaded-models");
        if (configured.isEmpty()) {
            return DEFAULT_MAX_LOADED_MODELS;
        }
        try {
            var maxLoadedModels = Integer.parseInt(String.valueOf(configured.get()).trim());
            if (maxLoadedModels >= 1) {
                return maxLoadedModels;
            }
        } catch (NumberFormatException e) {
            // reported below with the accepted values
        }
        throw new IllegalArgumentException("Plugin configuration `max-loaded-models` of " + getType() + " must be a whole number of at least 1, got `" + configured.get() + "`.");
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

    /**
     * What tasks receive. It holds no native resource: every call leases the model from {@link #MODELS},
     * loading it again from the task's working-dir copies if it was evicted since the previous call.
     */
    static final class CachedEmbeddingModel extends DimensionAwareEmbeddingModel {
        private final String cacheKey;
        private final int maxLoadedModels;
        private final Supplier<LoadedModel> loader;

        private CachedEmbeddingModel(String cacheKey, int maxLoadedModels, Supplier<LoadedModel> loader) {
            this.cacheKey = cacheKey;
            this.maxLoadedModels = maxLoadedModels;
            this.loader = loader;
        }

        // the loaded model remembers its dimension, so tasks do not each embed a probe text
        @Override
        public int dimension() {
            return MODELS.withModel(cacheKey, maxLoadedModels, loader, LoadedModel::dimension);
        }

        String cacheKey() {
            return cacheKey;
        }

        @Override
        public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
            return MODELS.withModel(cacheKey, maxLoadedModels, loader, model -> model.embedAll(segments));
        }
    }

    /**
     * One ONNX Runtime session, its tokenizer and the threads embedding segments in parallel. Unlike langchain4j's {@code OnnxEmbeddingModel}, it can be closed.
     */
    static final class LoadedModel extends AbstractInProcessEmbeddingModel implements AutoCloseable {
        private static final Duration CLOSE_TIMEOUT = Duration.ofSeconds(30);

        private final OnnxBertBiEncoder encoder;
        private final OrtSession session;
        private final ExecutorService executor;

        private LoadedModel(OnnxBertBiEncoder encoder, OrtSession session, ExecutorService executor) {
            super(executor);
            this.encoder = encoder;
            this.session = session;
            this.executor = executor;
        }

        static LoadedModel load(Path modelPath, Path tokenizerPath, PoolingMode poolingMode) {
            var environment = OrtEnvironment.getEnvironment();
            OrtSession session = null;
            try {
                session = environment.createSession(modelPath.toString());
                try (var tokenizer = Files.newInputStream(tokenizerPath)) {
                    return new LoadedModel(new OnnxBertBiEncoder(environment, session, tokenizer, poolingMode), session, newExecutor());
                }
            } catch (Exception e) {
                var failure = new IllegalArgumentException("Unable to load the ONNX model and tokenizer.", e);
                if (session != null) {
                    try {
                        session.close();
                    } catch (Exception closeFailure) {
                        failure.addSuppressed(closeFailure);
                    }
                }
                throw failure;
            }
        }

        // same sizing as langchain4j's default executor, but owned here so that close() can wait for the segments it runs
        private static ExecutorService newExecutor() {
            var threads = Runtime.getRuntime().availableProcessors();
            var executor = new ThreadPoolExecutor(
                threads, threads, 1, TimeUnit.SECONDS, new LinkedBlockingQueue<>(),
                Thread.ofPlatform().name("onnx-embedding-", 0).daemon().factory()
            );
            executor.allowCoreThreadTimeOut(true);
            return executor;
        }

        @Override
        protected OnnxBertBiEncoder model() {
            return encoder;
        }

        // embedAll returns as soon as one segment fails or the calling task is interrupted, while other segments may still be running:
        // closing the session under them crashes the worker, so it is closed only once they are done. The tokenizer frees its native memory on GC.
        @Override
        public void close() {
            executor.shutdownNow();
            if (!awaitSegmentsDone()) {
                throw new IllegalStateException("ONNX model segments still running after " + CLOSE_TIMEOUT.toSeconds() + "s, the model stays loaded.");
            }
            try {
                session.close();
            } catch (OrtException e) {
                throw new IllegalStateException("Unable to unload the ONNX model.", e);
            }
        }

        // keeps waiting when interrupted, since the closing thread is often the interrupted task itself
        private boolean awaitSegmentsDone() {
            var deadline = System.nanoTime() + CLOSE_TIMEOUT.toNanos();
            var interrupted = false;
            try {
                while (true) {
                    try {
                        return executor.awaitTermination(deadline - System.nanoTime(), TimeUnit.NANOSECONDS);
                    } catch (InterruptedException e) {
                        interrupted = true;
                    }
                }
            } finally {
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    /**
     * Keeps at most {@code maxLoadedModels} models loaded in the worker, unloading the least recently used idle one.
     * A model is never unloaded while an embedding call uses it, so each model has a single loaded copy;
     * when more distinct models are in use at once than the limit, the extra ones are unloaded as their calls return.
     */
    static final class ModelCache {
        private static final Logger log = LoggerFactory.getLogger(ModelCache.class);

        // accessOrder = true: iteration starts at the least recently used model
        private final LinkedHashMap<String, CachedModel> modelsByKey = new LinkedHashMap<>(16, 0.75f, true);

        <T> T withModel(String cacheKey, int maxLoadedModels, Supplier<LoadedModel> loader, Function<LoadedModel, T> action) {
            var cachedModel = startUsing(cacheKey, maxLoadedModels);
            try {
                return action.apply(loaded(cachedModel, loader));
            } finally {
                stopUsing(cachedModel, maxLoadedModels);
            }
        }

        synchronized boolean isLoaded(String cacheKey) {
            return modelsByKey.containsKey(cacheKey);
        }

        private CachedModel startUsing(String cacheKey, int maxLoadedModels) {
            CachedModel cachedModel;
            List<CachedModel> idleModels;
            synchronized (this) {
                cachedModel = modelsByKey.get(cacheKey);
                if (cachedModel == null) {
                    cachedModel = new CachedModel(cacheKey);
                    modelsByKey.put(cacheKey, cachedModel);
                }
                // counted first, so the model about to be used is never picked as idle
                cachedModel.activeUses++;
                idleModels = unloadLeastRecentlyUsed(maxLoadedModels);
            }
            // closed before the new model is loaded, so their memory is released first
            idleModels.forEach(ModelCache::close);
            return cachedModel;
        }

        // the first call to use a cache entry loads the model, while calls needing the same model wait for it instead of loading their own copy;
        // loading happens outside the lock so that calls using other models are not blocked
        private LoadedModel loaded(CachedModel cachedModel, Supplier<LoadedModel> loader) {
            if (cachedModel.loadStarted.compareAndSet(false, true)) {
                try {
                    cachedModel.loading.complete(loader.get());
                } catch (RuntimeException | Error e) {
                    synchronized (this) {
                        modelsByKey.remove(cachedModel.cacheKey, cachedModel);
                    }
                    cachedModel.loading.completeExceptionally(e);
                    throw e;
                }
            }
            try {
                return cachedModel.loading.join();
            } catch (CompletionException e) {
                throw e.getCause() instanceof RuntimeException loadFailure ? loadFailure : e;
            }
        }

        private void stopUsing(CachedModel cachedModel, int maxLoadedModels) {
            List<CachedModel> idleModels;
            synchronized (this) {
                cachedModel.activeUses--;
                idleModels = unloadLeastRecentlyUsed(maxLoadedModels);
            }
            idleModels.forEach(ModelCache::close);
        }

        // removes idle models, least recently used first, until the limit is met or only models in use remain;
        // returns them so the caller closes them outside the lock
        private List<CachedModel> unloadLeastRecentlyUsed(int maxLoadedModels) {
            var idleModels = new ArrayList<CachedModel>();
            var leastRecentlyUsedFirst = modelsByKey.values().iterator();
            while (modelsByKey.size() > maxLoadedModels && leastRecentlyUsedFirst.hasNext()) {
                var cachedModel = leastRecentlyUsedFirst.next();
                if (cachedModel.activeUses == 0) {
                    leastRecentlyUsedFirst.remove();
                    idleModels.add(cachedModel);
                }
            }
            return idleModels;
        }

        // a failure only leaks that model's memory, so it must not fail the call that happened to unload it
        private static void close(CachedModel cachedModel) {
            if (cachedModel.loading.state() != Future.State.SUCCESS) {
                return;
            }
            try {
                cachedModel.loading.resultNow().close();
            } catch (RuntimeException e) {
                log.warn("Unable to unload ONNX embedding model, its memory stays allocated until the worker restarts.", e);
            }
        }

        private static final class CachedModel {
            private final String cacheKey;
            private final AtomicBoolean loadStarted = new AtomicBoolean();
            private final CompletableFuture<LoadedModel> loading = new CompletableFuture<>();
            // guarded by the ModelCache lock
            private int activeUses;

            private CachedModel(String cacheKey) {
                this.cacheKey = cacheKey;
            }
        }
    }
}
