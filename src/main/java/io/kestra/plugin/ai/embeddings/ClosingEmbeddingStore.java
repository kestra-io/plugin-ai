package io.kestra.plugin.ai.embeddings;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;

/** An {@link EmbeddingStore} that owns a per-run resource, such as a client, and releases it on {@link #close()}. */
final class ClosingEmbeddingStore<T> implements EmbeddingStore<T>, AutoCloseable {
    private final EmbeddingStore<T> delegate;
    private final AutoCloseable resource;

    private ClosingEmbeddingStore(EmbeddingStore<T> delegate, AutoCloseable resource) {
        this.delegate = delegate;
        this.resource = resource;
    }

    /** Wraps the store built by {@code factory}, closing {@code resource} right away if building it fails. */
    static <T, E extends Exception> EmbeddingStore<T> of(AutoCloseable resource, StoreFactory<T, E> factory) throws E {
        try {
            return new ClosingEmbeddingStore<>(factory.create(), resource);
        } catch (Exception e) {
            try {
                resource.close();
            } catch (Exception closeError) {
                e.addSuppressed(closeError);
            }
            throw e;
        }
    }

    @Override
    public String add(Embedding embedding) {
        return delegate.add(embedding);
    }

    @Override
    public void add(String id, Embedding embedding) {
        delegate.add(id, embedding);
    }

    @Override
    public String add(Embedding embedding, T embedded) {
        return delegate.add(embedding, embedded);
    }

    @Override
    public List<String> addAll(List<Embedding> embeddings) {
        return delegate.addAll(embeddings);
    }

    @Override
    public List<String> addAll(List<Embedding> embeddings, List<T> embedded) {
        return delegate.addAll(embeddings, embedded);
    }

    @Override
    public void addAll(List<String> ids, List<Embedding> embeddings, List<T> embedded) {
        delegate.addAll(ids, embeddings, embedded);
    }

    @Override
    public List<String> generateIds(int n) {
        return delegate.generateIds(n);
    }

    @Override
    public void remove(String id) {
        delegate.remove(id);
    }

    @Override
    public void removeAll(Collection<String> ids) {
        delegate.removeAll(ids);
    }

    @Override
    public void removeAll(Filter filter) {
        delegate.removeAll(filter);
    }

    @Override
    public void removeAll() {
        delegate.removeAll();
    }

    @Override
    public EmbeddingSearchResult<T> search(EmbeddingSearchRequest request) {
        return delegate.search(request);
    }

    @Override
    public CompletableFuture<EmbeddingSearchResult<T>> searchAsync(EmbeddingSearchRequest request) {
        return delegate.searchAsync(request);
    }

    @Override
    public void close() throws Exception {
        resource.close();
    }

    @FunctionalInterface
    interface StoreFactory<T, E extends Exception> {
        EmbeddingStore<T> create() throws E;
    }
}
