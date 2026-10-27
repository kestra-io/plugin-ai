package io.kestra.plugin.ai.rag;

import java.util.concurrent.atomic.AtomicBoolean;

import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

/** An embedding store provider whose in-memory store records whether the task closed it. */
public final class ClosableEmbeddingStoreProvider extends EmbeddingStoreProvider {
    final AtomicBoolean closed = new AtomicBoolean();

    public ClosableEmbeddingStoreProvider() {
    }

    @Override
    public EmbeddingStore<TextSegment> embeddingStore(RunContext runContext, int dimension, boolean drop) {
        return new ClosableStore();
    }

    private class ClosableStore extends InMemoryEmbeddingStore<TextSegment> implements AutoCloseable {
        @Override
        public void close() {
            closed.set(true);
        }
    }
}
