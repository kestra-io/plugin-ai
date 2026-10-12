package io.kestra.plugin.ai.provider;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import io.kestra.plugin.ai.provider.Onnx.LoadedModel;
import io.kestra.plugin.ai.provider.Onnx.ModelCache;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.onnx.PoolingMode;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

// Uses the real all-MiniLM model, so "closed" means the native ONNX Runtime session was really closed.
class OnnxModelCacheTest {
    @TempDir
    static Path modelDir;

    private static Path modelFile;
    private static Path tokenizerFile;

    private final ModelCache cache = new ModelCache(Duration.ofMinutes(10));
    private final AtomicInteger loads = new AtomicInteger();

    @AfterEach
    void unloadModels() throws InterruptedException {
        cache.shutdown();
    }

    @BeforeAll
    static void copyModelFiles() throws Exception {
        modelFile = copyResource("all-minilm-l6-v2.onnx");
        tokenizerFile = copyResource("all-minilm-l6-v2-tokenizer.json");
    }

    @Test
    void reusesLoadedModelForSameKey() {
        var first = use("a", 2);
        var second = use("a", 2);

        assertThat(second).isSameAs(first);
        assertThat(loads).hasValue(1);
    }

    @Test
    void unloadsAndClosesLeastRecentlyUsedModel() {
        var a = use("a", 2);
        var b = use("b", 2);
        use("a", 2);

        var c = use("c", 2);

        assertClosed(b);
        assertThat(cache.isLoaded("b")).isFalse();
        assertUsable(a);
        assertUsable(c);
    }

    @Test
    void neverUnloadsAModelWhileACallUsesIt() {
        var b = new AtomicReference<LoadedModel>();
        var a = cache.withModel("a", 1, this::load, modelInUse ->
        {
            b.set(use("b", 1)); // over the limit, but "a" is in use: the idle "b" is unloaded instead
            assertUsable(modelInUse);
            assertThat(cache.isLoaded("a")).isTrue();
            return modelInUse;
        });

        assertClosed(b.get());
        assertUsable(a);
        assertThat(cache.isLoaded("a")).isTrue();
    }

    @Test
    void reusesTheLoadedCopyOfAModelInUseInsteadOfLoadingAnother() {
        cache.withModel("a", 1, this::load, modelInUse ->
        {
            use("b", 1);
            assertThat(use("a", 1)).isSameAs(modelInUse);
            return modelInUse;
        });

        assertThat(loads).hasValue(2);
    }

    @Test
    void reloadsUnloadedModelOnNextUse() {
        var a = use("a", 1);
        use("b", 1);

        var reloaded = use("a", 1);

        assertThat(reloaded).isNotSameAs(a);
        assertUsable(reloaded);
        assertThat(loads).hasValue(3);
    }

    @Test
    void loadsModelOnceWhenCallsFirstUseItConcurrently() throws Exception {
        var firstLoadStarted = new CountDownLatch(1);
        var releaseFirstLoad = new CountDownLatch(1);
        Supplier<LoadedModel> slowFirstLoad = () ->
        {
            if (loads.getAndIncrement() == 0) {
                firstLoadStarted.countDown();
                await(releaseFirstLoad);
            }
            return LoadedModel.load(modelFile, tokenizerFile, PoolingMode.MEAN);
        };

        try (var executor = Executors.newFixedThreadPool(2)) {
            var first = executor.submit(() -> cache.withModel("a", 2, slowFirstLoad, model -> model));
            assertThat(firstLoadStarted.await(30, TimeUnit.SECONDS)).isTrue();
            var secondThread = new AtomicReference<Thread>();
            var second = executor.submit(() ->
            {
                secondThread.set(Thread.currentThread());
                return cache.withModel("a", 2, slowFirstLoad, model -> model);
            });
            awaitDoneOrWaiting(second, secondThread);
            releaseFirstLoad.countDown();

            assertThat(second.get(30, TimeUnit.SECONDS)).isSameAs(first.get(30, TimeUnit.SECONDS));
        }
        assertThat(loads).hasValue(1);
    }

    @Test
    void failingToCloseAnUnloadedModelNeitherFailsNorLeaksTheCallUnloadingIt() {
        var a = use("a", 1);
        a.close(); // closing it again when it is unloaded fails

        var b = use("b", 1);
        assertUsable(b);

        // "b" is still unloaded and closed normally, so the failure left no use behind
        use("c", 1);
        assertClosed(b);
    }

    @Test
    void closesModelOnlyOnceSegmentsStillRunningFromAnInterruptedCallAreDone() throws Exception {
        var segments = IntStream.range(0, 400)
            .mapToObj(i -> TextSegment.from("Kestra orchestrates data pipelines and AI workflows. ".repeat(20) + i))
            .toList();

        // without waiting, the native session is freed under running segments and the JVM crashes within a few rounds
        for (int round = 0; round < 5; round++) {
            var model = load();
            var closeFailure = new AtomicReference<Throwable>();
            var interruptedCall = new Thread(() ->
            {
                try {
                    model.embedAll(segments);
                } catch (RuntimeException expected) {
                    // interrupted while waiting for the segments
                } finally {
                    try {
                        model.close();
                    } catch (Throwable e) {
                        closeFailure.set(e);
                    }
                }
            });
            interruptedCall.start();
            Thread.sleep(50);
            interruptedCall.interrupt();
            interruptedCall.join();

            assertThat(closeFailure.get()).isNull();
            assertClosed(model);
        }
    }

    @Test
    void loadsEveryModelOnOneThreadWhicheverThreadUsesIt() throws Exception {
        var loadingThreads = new CopyOnWriteArraySet<Thread>();
        Supplier<LoadedModel> recordingLoad = () ->
        {
            loadingThreads.add(Thread.currentThread());
            return load();
        };

        var callingThreads = new CopyOnWriteArraySet<Thread>();
        for (var cacheKey : List.of("a", "b", "c", "a")) {
            try (var executor = Executors.newSingleThreadExecutor()) {
                executor.submit(() ->
                {
                    callingThreads.add(Thread.currentThread());
                    return cache.withModel(cacheKey, 1, recordingLoad, model -> model);
                }).get(30, TimeUnit.SECONDS);
            }
        }

        assertThat(loads).hasValue(4);
        assertThat(loadingThreads).hasSize(1).doesNotContainAnyElementsOf(callingThreads);
    }

    @Test
    void interruptedCallStopsWaitingWhileTheLoadCompletesForTheNextCall() throws Exception {
        var releaseLoad = new CountDownLatch(1);
        Supplier<LoadedModel> slowLoad = () ->
        {
            await(releaseLoad);
            return load();
        };

        try (var executor = Executors.newSingleThreadExecutor()) {
            var waitingThread = new AtomicReference<Thread>();
            var stillInterrupted = new AtomicReference<Boolean>();
            var call = executor.submit(() ->
            {
                waitingThread.set(Thread.currentThread());
                try {
                    return cache.withModel("a", 1, slowLoad, model -> model);
                } finally {
                    stillInterrupted.set(Thread.currentThread().isInterrupted());
                }
            });
            awaitDoneOrWaiting(call, waitingThread);
            waitingThread.get().interrupt();

            assertThatThrownBy(() -> call.get(5, TimeUnit.SECONDS)).hasCauseInstanceOf(IllegalStateException.class);
            assertThat(stillInterrupted).hasValue(true);
        }

        releaseLoad.countDown();
        assertUsable(use("a", 1));
        assertThat(loads).hasValue(1);
    }

    @Test
    void failedLoadReachesTheCallerAndLaterLoadsStillWork() {
        var failure = new IllegalArgumentException("broken model");
        assertThatThrownBy(() -> cache.withModel("a", 1, () ->
        {
            throw failure;
        }, model -> model)).isSameAs(failure);
        var error = new NoClassDefFoundError("ai/onnxruntime/OrtSession");
        assertThatThrownBy(() -> cache.withModel("a", 1, () ->
        {
            throw error;
        }, model -> model)).isSameAs(error);
        assertThat(cache.isLoaded("a")).isFalse();

        assertUsable(use("a", 1));
        assertThat(cache.isLoaded("a")).isTrue();
    }

    @Test
    void concurrentCallsOnMoreModelsThanTheLimitAllSucceed() throws Exception {
        var keys = List.of("a", "b", "c");
        try (var executor = Executors.newFixedThreadPool(6)) {
            var calls = IntStream.range(0, 36)
                .mapToObj(i -> executor.submit(() -> cache.withModel(keys.get(i % keys.size()), 2, this::load, model -> model.embed("hello " + i))))
                .toList();
            for (var call : calls) {
                assertThat(call.get(120, TimeUnit.SECONDS).content().vector()).hasSize(384);
            }
        }

        assertThat(keys.stream().filter(cache::isLoaded)).hasSizeLessThanOrEqualTo(2);
    }

    @Test
    void tokenizerWithoutClsAndSepFailsToLoadWithATokenizerMismatch() throws Exception {
        var tokenizer = writeTokenizer("no-special-tokens.json", "null", "\"test\": 2000");

        assertThatThrownBy(() -> LoadedModel.load(modelFile, tokenizer, PoolingMode.MEAN))
            .isInstanceOf(Onnx.TokenizerMismatchException.class)
            .hasMessageContaining("tokenizer.json");
    }

    @Test
    void tokenizerLargerThanTheModelVocabularyFailsToLoadWithATokenizerMismatch() throws Exception {
        var tokenizer = writeTokenizer("larger-vocabulary.json", CLS_SEP_TEMPLATE, "\"[CLS]\": 101, \"[SEP]\": 102, \"test\": 2000, \"kestra\": 99999");

        assertThatThrownBy(() -> LoadedModel.load(modelFile, tokenizer, PoolingMode.MEAN))
            .isInstanceOf(Onnx.TokenizerMismatchException.class)
            .hasMessageContaining("outside the ONNX model's vocabulary");
    }

    @Test
    void tokenizerFarSmallerThanTheModelVocabularyLoadsWithAWarning() throws Exception {
        var tokenizer = writeTokenizer("smaller-vocabulary.json", CLS_SEP_TEMPLATE, "\"[CLS]\": 101, \"[SEP]\": 102, \"test\": 2000");
        var model = LoadedModel.load(modelFile, tokenizer, PoolingMode.MEAN);
        try {
            assertThat(model.vocabularyWarning()).hasValueSatisfying(warning -> assertThat(warning).contains("not the one published with this model"));
        } finally {
            model.close();
        }
    }

    @Test
    void publishedTokenizerLoadsWithoutVocabularyWarning() {
        var model = load();
        try {
            assertThat(model.vocabularyWarning()).isEmpty();
        } finally {
            model.close();
        }
    }

    @Test
    void unloadsAndClosesAModelUnusedForTheIdleTimeout() throws Exception {
        var shortIdleCache = new ModelCache(Duration.ofMillis(300));
        try {
            var model = shortIdleCache.withModel("a", 2, this::load, loaded -> loaded);

            awaitUntil(() -> !shortIdleCache.isLoaded("a"));
            shortIdleCache.awaitQueuedLoadsAndCloses();
            assertThatThrownBy(() -> model.embedAll(List.of(TextSegment.from("hello")))).isInstanceOf(IllegalStateException.class);
        } finally {
            shortIdleCache.shutdown();
        }
    }

    @Test
    void neverUnloadsAModelInUseForLongerThanTheIdleTimeout() throws Exception {
        var shortIdleCache = new ModelCache(Duration.ofMillis(300));
        try {
            shortIdleCache.withModel("a", 2, this::load, model ->
            {
                sleep(Duration.ofSeconds(1));
                assertThat(shortIdleCache.isLoaded("a")).isTrue();
                assertUsable(model);
                return model;
            });
        } finally {
            shortIdleCache.shutdown();
        }
    }

    @Test
    void waitingCallLoadsFromItsOwnFilesWhenTheSharedLoadFails() throws Exception {
        var firstLoadStarted = new CountDownLatch(1);
        var releaseFirstLoad = new CountDownLatch(1);
        var filesGone = new IllegalArgumentException("Unable to load the ONNX model: the first task's working directory was deleted");
        Supplier<LoadedModel> loadFromDeletedFiles = () ->
        {
            firstLoadStarted.countDown();
            await(releaseFirstLoad);
            throw filesGone;
        };

        try (var executor = Executors.newFixedThreadPool(2)) {
            var first = executor.submit(() -> cache.withModel("a", 2, loadFromDeletedFiles, model -> model));
            assertThat(firstLoadStarted.await(30, TimeUnit.SECONDS)).isTrue();
            var secondThread = new AtomicReference<Thread>();
            var second = executor.submit(() ->
            {
                secondThread.set(Thread.currentThread());
                return cache.withModel("a", 2, this::load, model -> model);
            });
            awaitDoneOrWaiting(second, secondThread);
            releaseFirstLoad.countDown();

            assertThatThrownBy(() -> first.get(30, TimeUnit.SECONDS)).hasCause(filesGone);
            assertUsable(second.get(30, TimeUnit.SECONDS));
        }
    }

    private static final String CLS_SEP_TEMPLATE = """
        {"type": "TemplateProcessing",
         "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
         "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                  {"Sequence": {"id": "B", "type_id": 1}}, {"SpecialToken": {"id": "[SEP]", "type_id": 1}}],
         "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]}, "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}}}""";

    private static Path writeTokenizer(String name, String postProcessor, String vocabulary) throws Exception {
        return Files.writeString(modelDir.resolve(name), """
            {"version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
             "pre_tokenizer": {"type": "Whitespace"}, "post_processor": %s, "decoder": null,
             "model": {"type": "WordLevel", "vocab": {"[UNK]": 100, %s}, "unk_token": "[UNK]"}}
            """.formatted(postProcessor, vocabulary));
    }

    private static void awaitUntil(BooleanSupplier condition) throws InterruptedException {
        var deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) {
            Thread.sleep(20);
        }
        assertThat(condition.getAsBoolean()).isTrue();
    }

    private static void sleep(Duration duration) {
        try {
            Thread.sleep(duration);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }

    private LoadedModel use(String cacheKey, int maxLoadedModels) {
        return cache.withModel(cacheKey, maxLoadedModels, this::load, model -> model);
    }

    private LoadedModel load() {
        loads.incrementAndGet();
        return LoadedModel.load(modelFile, tokenizerFile, PoolingMode.MEAN);
    }

    private static void assertUsable(LoadedModel model) {
        assertThat(model.embed("hello").content().vector()).hasSize(384);
    }

    private void assertClosed(LoadedModel model) {
        cache.awaitQueuedLoadsAndCloses();
        assertThatThrownBy(() -> model.embedAll(List.of(TextSegment.from("hello")))).isInstanceOf(IllegalStateException.class);
    }

    private static void awaitDoneOrWaiting(Future<?> call, AtomicReference<Thread> thread) throws InterruptedException {
        var deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (!call.isDone() && !isWaiting(thread.get()) && System.nanoTime() < deadline) {
            Thread.sleep(10);
        }
    }

    private static boolean isWaiting(Thread thread) {
        return thread != null && thread.getState() == Thread.State.WAITING;
    }

    private static void await(CountDownLatch latch) {
        try {
            assertThat(latch.await(30, TimeUnit.SECONDS)).isTrue();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }

    private static Path copyResource(String name) throws Exception {
        var target = modelDir.resolve(name);
        try (var in = OnnxModelCacheTest.class.getClassLoader().getResourceAsStream(name)) {
            Files.copy(in, target);
        }
        return target;
    }
}
