package io.kestra.plugin.ai.observability;

import dev.langchain4j.model.chat.listener.ChatModelErrorContext;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.listener.ChatModelRequestContext;
import dev.langchain4j.model.chat.listener.ChatModelResponseContext;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Context;
import org.slf4j.Logger;

/**
 * ChatModelListener that creates child OTel spans for each LLM request/response
 * within an agent invocation. Only used when Langfuse observability is enabled.
 */
public final class LangfuseChatModelListener implements ChatModelListener {
    private static final String SPAN_KEY = "kestra.langfuse.child.span";

    private final Tracer tracer;
    private final Logger logger;
    // The parent span is set after the main observability class creates it in onStart
    private volatile Span parentSpan;

    public LangfuseChatModelListener(Tracer tracer, Logger logger) {
        this.tracer = tracer;
        this.logger = logger;
    }

    void setParentSpan(Span parentSpan) {
        this.parentSpan = parentSpan;
    }

    @Override
    public void onRequest(ChatModelRequestContext requestContext) {
        try {
            if (parentSpan == null) {
                return;
            }
            var parentContext = Context.current().with(parentSpan);
            var childSpan = tracer.spanBuilder("gen_ai.chat")
                .setParent(parentContext)
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();

            var request = requestContext.chatRequest();
            if (request != null) {
                if (request.modelName() != null) {
                    childSpan.setAttribute("gen_ai.request.model", request.modelName());
                }
                if (request.temperature() != null) {
                    childSpan.setAttribute("gen_ai.request.temperature", request.temperature());
                }
                if (request.topP() != null) {
                    childSpan.setAttribute("gen_ai.request.top_p", request.topP());
                }
                if (request.maxOutputTokens() != null) {
                    childSpan.setAttribute("gen_ai.request.max_tokens", request.maxOutputTokens());
                }
            }

            requestContext.attributes().put(SPAN_KEY, childSpan);
        } catch (Exception e) {
            logger.debug("Failed to start Langfuse child span for chat model request", e);
        }
    }

    @Override
    public void onResponse(ChatModelResponseContext responseContext) {
        try {
            var childSpan = (Span) responseContext.attributes().get(SPAN_KEY);
            if (childSpan == null) {
                return;
            }

            var response = responseContext.chatResponse();
            if (response != null) {
                if (response.modelName() != null) {
                    childSpan.setAttribute("gen_ai.response.model", response.modelName());
                }
                if (response.tokenUsage() != null) {
                    var usage = response.tokenUsage();
                    if (usage.inputTokenCount() != null) {
                        childSpan.setAttribute("gen_ai.usage.input_tokens", usage.inputTokenCount());
                    }
                    if (usage.outputTokenCount() != null) {
                        childSpan.setAttribute("gen_ai.usage.output_tokens", usage.outputTokenCount());
                    }
                }
                if (response.finishReason() != null) {
                    childSpan.setAttribute("gen_ai.response.finish_reasons", response.finishReason().name());
                }
            }

            childSpan.setStatus(StatusCode.OK);
            childSpan.end();
        } catch (Exception e) {
            logger.debug("Failed to end Langfuse child span for chat model response", e);
        }
    }

    @Override
    public void onError(ChatModelErrorContext errorContext) {
        try {
            var childSpan = (Span) errorContext.attributes().get(SPAN_KEY);
            if (childSpan == null) {
                return;
            }

            if (errorContext.error() != null) {
                childSpan.recordException(errorContext.error());
                childSpan.setStatus(StatusCode.ERROR, errorContext.error().getMessage());
            } else {
                childSpan.setStatus(StatusCode.ERROR);
            }
            childSpan.end();
        } catch (Exception e) {
            logger.debug("Failed to handle Langfuse child span error", e);
        }
    }
}
