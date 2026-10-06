package io.kestra.plugin.ai.provider;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import java.util.stream.IntStream;

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

    private final ModelCache cache = new ModelCache();
    private final AtomicInteger loads = new AtomicInteger();

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

    private static void assertClosed(LoadedModel model) {
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
