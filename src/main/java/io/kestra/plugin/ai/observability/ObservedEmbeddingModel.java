package io.kestra.plugin.ai.observability;

import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.ContentType;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.listener.EmbeddingModelListener;
import dev.langchain4j.model.embedding.request.EmbeddingParameter;
import dev.langchain4j.model.embedding.request.EmbeddingRequest;
import dev.langchain4j.model.embedding.request.EmbeddingRequestParameters;
import dev.langchain4j.model.embedding.response.EmbeddingResponse;
import dev.langchain4j.model.output.Response;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;

final class ObservedEmbeddingModel implements EmbeddingModel {
    private final EmbeddingModel delegate;
    private final TelemetrySession session;

    ObservedEmbeddingModel(EmbeddingModel delegate, TelemetrySession session) {
        this.delegate = delegate;
        this.session = session;
    }

    @Override
    public EmbeddingResponse embed(EmbeddingRequest request) {
        var model = GenAiSpans.modelOrFallback(requestModel(request), session.fallbackModel);
        Span span = null;
        try {
            span = GenAiSpans.start(session, "embeddings", model);
            if (session.captureMessageContent && request != null && request.inputs() != null) {
                var texts = request.inputs().stream().map(input -> input.text() == null ? "" : input.text()).toList();
                var joined = String.join("\n", texts);
                GenAiSpans.put(span, GenAiSpans.INPUT_MESSAGES, GenAiMessages.textMessages("user", joined));
            }
        } catch (Exception e) {
            session.logger.debug("Failed to start an embedding span", e);
            GenAiSpans.end(span);
            return delegate.embed(request);
        }

        try (Scope ignored = span.makeCurrent()) {
            var response = delegate.embed(request);
            complete(span, response);
            return response;
        } catch (Exception e) {
            GenAiSpans.fail(session, span, e);
            span = null;
            if (e instanceof RuntimeException runtime) {
                throw runtime;
            }
            throw new IllegalStateException(e);
        } finally {
            GenAiSpans.end(span);
        }
    }

    @Override
    public CompletableFuture<EmbeddingResponse> doEmbedAsync(EmbeddingRequest request) {
        return delegate.doEmbedAsync(request);
    }

    @Override
    public Response<Embedding> embed(String text) {
        return EmbeddingModel.super.embed(text);
    }

    @Override
    public Response<Embedding> embed(TextSegment textSegment) {
        return EmbeddingModel.super.embed(textSegment);
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
        return EmbeddingModel.super.embedAll(textSegments);
    }

    @Override
    public int dimension() {
        return delegate.dimension();
    }

    @Override
    public String modelName() {
        return delegate.modelName();
    }

    @Override
    public EmbeddingRequestParameters defaultRequestParameters() {
        return delegate.defaultRequestParameters();
    }

    @Override
    public Set<EmbeddingParameter<?>> supportedParameters() {
        return delegate.supportedParameters();
    }

    @Override
    public Set<ContentType> supportedContentTypes() {
        return delegate.supportedContentTypes();
    }

    @Override
    public List<EmbeddingModelListener> listeners() {
        return delegate.listeners();
    }

    private static String requestModel(EmbeddingRequest request) {
        if (request == null || request.parameters() == null) {
            return null;
        }
        return request.parameters().modelName();
    }

    private void complete(Span span, EmbeddingResponse response) {
        try {
            if (response == null) {
                return;
            }
            GenAiSpans.put(span, GenAiSpans.RESPONSE_MODEL, response.modelName());
            if (response.tokenUsage() != null) {
                GenAiSpans.put(span, GenAiSpans.INPUT_TOKENS, response.tokenUsage().inputTokenCount());
                GenAiSpans.put(span, GenAiSpans.OUTPUT_TOKENS, response.tokenUsage().outputTokenCount());
            }
            if (response.embeddings() != null && !response.embeddings().isEmpty()) {
                span.setAttribute(GenAiSpans.EMBEDDING_DIMENSIONS, (long) response.embeddings().getFirst().dimension());
            }
        } catch (Exception e) {
            session.logger.debug("Failed to complete an embedding span", e);
        }
    }
}
