package io.kestra.plugin.ai.retriever;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.store.embedding.EmbeddingStore;

/**
 * A {@link ContentRetriever} that closes the embedding store it reads from on {@link #close()}.
 * Tasks close any retriever that is {@link AutoCloseable} once the run ends.
 */
public final class ClosingContentRetriever implements ContentRetriever, AutoCloseable {
    private final ContentRetriever delegate;
    private final AutoCloseable store;

    private ClosingContentRetriever(ContentRetriever delegate, AutoCloseable store) {
        this.delegate = delegate;
        this.store = store;
    }

    /**
     * Returns {@code retriever} itself when {@code store} holds nothing to close.
     */
    public static ContentRetriever of(ContentRetriever retriever, EmbeddingStore<?> store) {
        return store instanceof AutoCloseable closeable ? new ClosingContentRetriever(retriever, closeable) : retriever;
    }

    @Override
    public List<Content> retrieve(Query query) {
        return delegate.retrieve(query);
    }

    @Override
    public CompletableFuture<List<Content>> retrieveAsync(Query query) {
        return delegate.retrieveAsync(query);
    }

    @Override
    public void close() throws Exception {
        store.close();
    }
}
