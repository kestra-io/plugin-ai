package io.kestra.plugin.ai.provider;

import java.io.IOException;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import java.util.function.Supplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.property.URIFetcher;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtLoggingLevel;
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
        Provide the `.onnx` model file and its Hugging Face `tokenizer.json`; `modelName` is only used as a label, and `baseUrl`, `clientPem` and `caPem` are ignored since nothing is called over the network. \
        Embeddings only: chat completion and image generation are not supported. \
        A loaded model stays in the worker's memory and is reused by every task that provides the same model and tokenizer files; it is unloaded after about 10 minutes without use. \
        Each worker keeps at most 2 models loaded between embedding calls (each distinct model, tokenizer and `poolingMode` combination counts as one): loading another one unloads the least recently used model as soon as no running call uses it, and that model is loaded again on its next use. \
        Administrators can change the limit with the `max-loaded-models` plugin configuration of `io.kestra.plugin.ai.provider.Onnx`; set it to at least the number of distinct models used at the same time, otherwise they keep unloading and reloading each other. \
        Models are loaded and unloaded on a single worker thread so that reloading them reuses the same native memory: a worker loads one model at a time, so loading a large model delays the first use of another model, while calls using already loaded models are not affected. \
        On glibc-based workers that also run other plugins allocating native memory from many threads, setting the `MALLOC_ARENA_MAX=2` environment variable on the worker further limits native memory growth."""
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
    private static final Duration MODEL_IDLE_TIMEOUT = Duration.ofMinutes(10);

    // ONNX Runtime sessions hold native memory that only an explicit close() releases, so the plugin owns them and bounds how many stay loaded.
    static final ModelCache MODELS = new ModelCache(MODEL_IDLE_TIMEOUT);

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
        description = "URI of the Hugging Face `tokenizer.json` published with the model, using the same schemes as `modelUri`. A tokenizer from another model fails, or silently produces wrong embeddings when its token ids happen to fit the model's vocabulary.",
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
                var model = LoadedModel.load(modelFile.path(), tokenizerFile.path(), rPoolingMode);
                model.vocabularyWarning().ifPresent(warning -> runContext.logger().warn(warning));
                return model;
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

    // holds no native resource: each call leases the model, reloading it from the task's working-dir copies if evicted
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

    static final class TokenizerMismatchException extends IllegalArgumentException {
        private TokenizerMismatchException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    // unlike langchain4j's OnnxEmbeddingModel, it can be closed
    static final class LoadedModel extends AbstractInProcessEmbeddingModel implements AutoCloseable {
        private static final Duration CLOSE_TIMEOUT = Duration.ofSeconds(30);
        private static final String OUTSIDE_VOCABULARY = "The tokenizer produced token ids outside the ONNX model's vocabulary: use the `tokenizer.json` published with the model.";
        private static final Set<String> PROBED_INPUTS = Set.of("input_ids", "attention_mask", "token_type_ids");
        // exported models may round their vocabulary up, e.g. to a multiple of 64, beyond the tokenizer's
        private static final int VOCABULARY_PADDING = 128;

        private final OnnxBertBiEncoder encoder;
        private final OrtSession session;
        private final ExecutorService executor;
        private String vocabularyWarning;

        private LoadedModel(OnnxBertBiEncoder encoder, OrtSession session, ExecutorService executor) {
            super(executor);
            this.encoder = encoder;
            this.session = session;
            this.executor = executor;
        }

        static LoadedModel load(Path modelPath, Path tokenizerPath, PoolingMode poolingMode) {
            var model = open(modelPath, tokenizerPath, poolingMode);
            try {
                model.checkVocabulary(tokenizerPath);
                // embeds a probe text: langchain4j drops the first and last token, so a tokenizer without [CLS] and [SEP] fails here
                model.dimension();
                return model;
            } catch (RuntimeException | Error e) {
                Throwable failure = e instanceof RuntimeException && !(e instanceof TokenizerMismatchException) ? new TokenizerMismatchException(
                    "The ONNX model cannot embed a test text with this tokenizer: use a BERT-style sentence-embedding model and the `tokenizer.json` published with it.", e
                ) : e;
                try {
                    model.close();
                } catch (RuntimeException closeFailure) {
                    failure.addSuppressed(closeFailure);
                }
                if (failure instanceof Error error) {
                    throw error;
                }
                throw (RuntimeException) failure;
            }
        }

        private static LoadedModel open(Path modelPath, Path tokenizerPath, PoolingMode poolingMode) {
            var environment = OrtEnvironment.getEnvironment();
            OrtSession session;
            try {
                session = environment.createSession(modelPath.toString());
            } catch (OrtException e) {
                throw new IllegalArgumentException("Unable to load the ONNX model: check that `modelUri` points to a valid `.onnx` file.", e);
            }
            try (var tokenizer = Files.newInputStream(tokenizerPath)) {
                return new LoadedModel(new OnnxBertBiEncoder(environment, session, tokenizer, poolingMode), session, newExecutor());
            } catch (Exception e) {
                var failure = new IllegalArgumentException("Unable to read the tokenizer: check that `tokenizerUri` points to the Hugging Face `tokenizer.json` of the model.", e);
                closeAfterFailure(session, failure);
                throw failure;
            } catch (Error e) {
                closeAfterFailure(session, e);
                throw e;
            }
        }

        private static void closeAfterFailure(OrtSession session, Throwable failure) {
            try {
                session.close();
            } catch (Exception closeFailure) {
                failure.addSuppressed(closeFailure);
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

        Optional<String> vocabularyWarning() {
            return Optional.ofNullable(vocabularyWarning);
        }

        // a tokenizer from another model either produces ids beyond the model's vocabulary, or leaves a large part of it unused
        private void checkVocabulary(Path tokenizerPath) {
            var tokenizerVocabulary = tokenizerVocabularySize(tokenizerPath);
            if (tokenizerVocabulary.isEmpty() || !PROBED_INPUTS.containsAll(session.getInputNames())) {
                return;
            }
            var size = tokenizerVocabulary.getAsInt();
            if (!acceptsTokenId(size - 1)) {
                throw new TokenizerMismatchException(OUTSIDE_VOCABULARY, null);
            }
            if (acceptsTokenId(size - 1L + VOCABULARY_PADDING)) {
                vocabularyWarning = "The ONNX model accepts more than " + VOCABULARY_PADDING + " token ids beyond the " + size + " of the tokenizer: "
                    + "the tokenizer is probably not the one published with this model, which silently produces wrong embeddings.";
            }
        }

        private boolean acceptsTokenId(long tokenId) {
            var environment = OrtEnvironment.getEnvironment();
            long[] shape = { 1, 1 };
            try (
                var ids = OnnxTensor.createTensor(environment, LongBuffer.wrap(new long[] { tokenId }), shape);
                var mask = OnnxTensor.createTensor(environment, LongBuffer.wrap(new long[] { 1 }), shape);
                var types = OnnxTensor.createTensor(environment, LongBuffer.wrap(new long[] { 0 }), shape);
                var runOptions = new OrtSession.RunOptions()
            ) {
                // an id beyond the vocabulary is an expected answer here, not an error for ONNX Runtime to log
                runOptions.setLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL);
                var tensors = Map.of("input_ids", ids, "attention_mask", mask, "token_type_ids", types);
                var inputs = new HashMap<String, OnnxTensor>();
                session.getInputNames().forEach(name -> inputs.put(name, tensors.get(name)));
                session.run(inputs, runOptions).close();
                return true;
            } catch (OrtException e) {
                if (isOutsideVocabulary(e)) {
                    return false;
                }
                throw new IllegalStateException(e);
            }
        }

        // the largest id among the vocabulary and the added tokens, plus one
        private static OptionalInt tokenizerVocabularySize(Path tokenizerPath) {
            JsonNode tokenizer;
            try {
                tokenizer = JacksonMapper.ofJson().readTree(tokenizerPath.toFile());
            } catch (IOException e) {
                return OptionalInt.empty();
            }
            var vocabulary = tokenizer.path("model").path("vocab");
            var largestId = vocabulary.isArray() ? vocabulary.size() - 1 : -1;
            if (vocabulary.isObject()) {
                for (var id : vocabulary) {
                    largestId = Math.max(largestId, id.asInt(-1));
                }
            }
            for (var addedToken : tokenizer.path("added_tokens")) {
                largestId = Math.max(largestId, addedToken.path("id").asInt(-1));
            }
            return largestId < 0 ? OptionalInt.empty() : OptionalInt.of(largestId + 1);
        }

        @Override
        public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
            try {
                return super.embedAll(segments);
            } catch (RuntimeException e) {
                if (isOutsideVocabulary(e)) {
                    throw new TokenizerMismatchException(OUTSIDE_VOCABULARY, e);
                }
                throw e;
            }
        }

        private static boolean isOutsideVocabulary(Throwable failure) {
            for (var cause = failure; cause != null; cause = cause.getCause()) {
                if (cause instanceof OrtException && cause.getMessage() != null && cause.getMessage().contains("out of data bounds")) {
                    return true;
                }
            }
            return false;
        }

        // embedAll returns as soon as one segment fails or the calling task is interrupted, while other segments may still be running:
        // closing the session under them crashes the worker, so it is closed only once they are done
        @Override
        public void close() {
            executor.shutdownNow();
            if (!awaitSegmentsDone()) {
                throw new IllegalStateException("ONNX model segments still running after " + CLOSE_TIMEOUT.toSeconds() + "s, the model stays loaded.");
            }
            RuntimeException failure = null;
            try {
                session.close();
            } catch (OrtException e) {
                failure = new IllegalStateException("Unable to unload the ONNX model.", e);
            } catch (RuntimeException e) {
                failure = e;
            }
            try {
                closeTokenizer();
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
            if (failure != null) {
                throw failure;
            }
        }

        // langchain4j's encoder neither closes nor exposes its tokenizer, whose native memory would otherwise wait for a GC to run its finalizer
        private void closeTokenizer() {
            try {
                var tokenizer = OnnxBertBiEncoder.class.getDeclaredField("tokenizer");
                tokenizer.setAccessible(true);
                ((AutoCloseable) tokenizer.get(encoder)).close();
            } catch (Exception e) {
                throw new IllegalStateException("Unable to unload the ONNX tokenizer.", e);
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

    // a model is never unloaded while a call uses it, so the limit is exceeded until those calls return
    static final class ModelCache {
        private static final Logger log = LoggerFactory.getLogger(ModelCache.class);
        private static final Duration LONG_QUEUE_WAIT = Duration.ofSeconds(5);

        // accessOrder = true: iteration starts at the least recently used model
        private final LinkedHashMap<String, CachedModel> modelsByKey = new LinkedHashMap<>(16, 0.75f, true);
        private final Duration idleTimeout;

        // glibc keeps a malloc arena per allocating thread: loading and closing every model on one thread that lives as long as
        // the worker keeps their native memory in a single arena, instead of spreading it over the worker threads running tasks
        private final ScheduledThreadPoolExecutor lifecycle;

        // guarded by this
        private ScheduledFuture<?> idleModelsSweep;

        // not inheriting the caller's context class loader lets an uninstalled plugin's classes be unloaded; withPluginClassLoader
        // lends the plugin's only while the thread works
        ModelCache(Duration idleTimeout) {
            this.idleTimeout = idleTimeout;
            this.lifecycle = new ScheduledThreadPoolExecutor(1, Thread.ofPlatform().name("onnx-model-lifecycle").daemon().inheritInheritableThreadLocals(false).factory());
            lifecycle.setRemoveOnCancelPolicy(true);
        }

        <T> T withModel(String cacheKey, int maxLoadedModels, Supplier<LoadedModel> loader, Function<LoadedModel, T> action) {
            try {
                return useModel(cacheKey, maxLoadedModels, loader, action, true);
            } catch (SharedLoadFailedException e) {
                // the call that started the load may have ended and taken its working-dir files with it: load from a running call's files
                return useModel(cacheKey, maxLoadedModels, loader, action, false);
            }
        }

        private <T> T useModel(String cacheKey, int maxLoadedModels, Supplier<LoadedModel> loader, Function<LoadedModel, T> action, boolean retryOwnLoadOnSharedFailure) {
            var cachedModel = startUsing(cacheKey, maxLoadedModels);
            try {
                return action.apply(loaded(cachedModel, loader, retryOwnLoadOnSharedFailure));
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
                    startIdleModelsSweep();
                    cachedModel = new CachedModel(cacheKey);
                    modelsByKey.put(cacheKey, cachedModel);
                }
                // counted first, so the model about to be used is never picked as idle
                cachedModel.activeUses++;
                idleModels = unloadLeastRecentlyUsed(maxLoadedModels);
            }
            // queued behind earlier loads, so their memory is released before the new model is loaded
            close(idleModels, "it is the least recently used one beyond the max-loaded-models limit");
            return cachedModel;
        }

        // guarded by this; started with the first loaded model and stopped by the sweep once none is left
        private void startIdleModelsSweep() {
            if (idleModelsSweep == null) {
                var period = idleTimeout.dividedBy(2).toNanos();
                idleModelsSweep = lifecycle.scheduleWithFixedDelay(withPluginClassLoader(this::unloadIdleModels), period, period, TimeUnit.NANOSECONDS);
            }
        }

        // the first call to use a cache entry loads the model, while calls needing the same model wait for it instead of loading their own copy;
        // loading happens outside the lock so that calls using already loaded models are not blocked
        private LoadedModel loaded(CachedModel cachedModel, Supplier<LoadedModel> loader, boolean retryOwnLoadOnSharedFailure) {
            var startedHere = cachedModel.loadStarted.compareAndSet(false, true);
            if (startedHere) {
                lifecycle.execute(onLifecycleThread("load ONNX embedding model " + cachedModel.cacheKey, () -> load(cachedModel, loader)));
            }
            try {
                return cachedModel.loading.get();
            } catch (InterruptedException e) {
                // a killed task stops waiting, and the load goes on so that the next call uses the model
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Interrupted while waiting for the ONNX embedding model to load.", e);
            } catch (ExecutionException e) {
                if (!startedHere && retryOwnLoadOnSharedFailure) {
                    throw new SharedLoadFailedException();
                }
                if (e.getCause() instanceof RuntimeException loadFailure) {
                    throw loadFailure;
                }
                if (e.getCause() instanceof Error loadError) {
                    throw loadError;
                }
                throw new IllegalStateException(e.getCause());
            }
        }

        private void load(CachedModel cachedModel, Supplier<LoadedModel> loader) {
            try {
                cachedModel.loading.complete(loader.get());
            } catch (RuntimeException | Error e) {
                synchronized (this) {
                    modelsByKey.remove(cachedModel.cacheKey, cachedModel);
                }
                cachedModel.loading.completeExceptionally(e);
            }
        }

        private void stopUsing(CachedModel cachedModel, int maxLoadedModels) {
            List<CachedModel> idleModels;
            synchronized (this) {
                if (--cachedModel.activeUses == 0) {
                    cachedModel.idleSince = System.nanoTime();
                }
                idleModels = unloadLeastRecentlyUsed(maxLoadedModels);
            }
            close(idleModels, "it is the least recently used one beyond the max-loaded-models limit");
        }

        // the caller does not wait: the lifecycle thread runs closes and loads in order, so a later load still comes after these closes
        private void close(List<CachedModel> idleModels, String reason) {
            if (!idleModels.isEmpty()) {
                lifecycle.execute(onLifecycleThread("unload " + idleModels.size() + " ONNX embedding model(s)", () -> idleModels.forEach(cachedModel -> close(cachedModel, reason))));
            }
        }

        // the queue makes slow loads and closes of one model visible as delays for every other one, so long waits are logged
        private static Runnable onLifecycleThread(String work, Runnable task) {
            var queuedAt = System.nanoTime();
            return withPluginClassLoader(() ->
            {
                var waited = Duration.ofNanos(System.nanoTime() - queuedAt);
                if (waited.compareTo(LONG_QUEUE_WAIT) > 0) {
                    log.info("Waited {}s to {}: the worker loads and unloads one ONNX embedding model at a time.", waited.toSeconds(), work);
                }
                task.run();
            });
        }

        // DJL finds its native tokenizer library through the context class loader
        private static Runnable withPluginClassLoader(Runnable work) {
            return () ->
            {
                var thread = Thread.currentThread();
                var idleClassLoader = thread.getContextClassLoader();
                thread.setContextClassLoader(ModelCache.class.getClassLoader());
                try {
                    work.run();
                } finally {
                    thread.setContextClassLoader(idleClassLoader);
                }
            };
        }

        // for tests: waits for the loads and closes queued so far
        void awaitQueuedLoadsAndCloses() {
            CompletableFuture.runAsync(() ->
            {
            }, lifecycle).join();
        }

        // for tests: unloads every model, none of which may be in use, and stops the lifecycle thread
        void shutdown() throws InterruptedException {
            List<CachedModel> cachedModels;
            synchronized (this) {
                cachedModels = new ArrayList<>(modelsByKey.values());
                modelsByKey.clear();
            }
            lifecycle.execute(withPluginClassLoader(() -> cachedModels.forEach(cachedModel -> close(cachedModel, "the cache is shut down"))));
            lifecycle.shutdown();
            lifecycle.awaitTermination(1, TimeUnit.MINUTES);
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

        // runs on the lifecycle thread; also how models of an uninstalled plugin, which nothing calls anymore, get unloaded
        private void unloadIdleModels() {
            var idleModels = new ArrayList<CachedModel>();
            synchronized (this) {
                var now = System.nanoTime();
                var cachedModels = modelsByKey.values().iterator();
                while (cachedModels.hasNext()) {
                    var cachedModel = cachedModels.next();
                    if (cachedModel.activeUses == 0 && cachedModel.loading.isDone() && now - cachedModel.idleSince >= idleTimeout.toNanos()) {
                        cachedModels.remove();
                        idleModels.add(cachedModel);
                    }
                }
            }
            idleModels.forEach(cachedModel -> close(cachedModel, "it has not been used for " + idleTimeout.toMinutes() + " minutes"));
            synchronized (this) {
                if (modelsByKey.isEmpty() && idleModelsSweep != null) {
                    idleModelsSweep.cancel(false);
                    idleModelsSweep = null;
                }
            }
        }

        // a failure only leaks that model's memory; catching errors too keeps the lifecycle thread, and so its single arena, alive
        private void close(CachedModel cachedModel, String reason) {
            if (cachedModel.loading.state() != Future.State.SUCCESS) {
                return;
            }
            log.info("Unloading ONNX embedding model {}: {}.", cachedModel.cacheKey, reason);
            try {
                cachedModel.loading.resultNow().close();
            } catch (RuntimeException | Error e) {
                log.warn("Unable to unload ONNX embedding model, its memory stays allocated until the worker restarts.", e);
            }
        }

        private static final class SharedLoadFailedException extends RuntimeException {
            private SharedLoadFailedException() {
                super(null, null, false, false);
            }
        }

        private static final class CachedModel {
            private final String cacheKey;
            private final AtomicBoolean loadStarted = new AtomicBoolean();
            private final CompletableFuture<LoadedModel> loading = new CompletableFuture<>();
            // guarded by the ModelCache lock
            private int activeUses;
            private long idleSince;

            private CachedModel(String cacheKey) {
                this.cacheKey = cacheKey;
            }
        }
    }
}
