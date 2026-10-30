package io.kestra.plugin.ai.observability;

import java.util.List;

import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.propagation.TextMapGetter;

final class GenAiSpans {
    static final AttributeKey<String> OPERATION = AttributeKey.stringKey("gen_ai.operation.name");
    static final AttributeKey<String> PROVIDER = AttributeKey.stringKey("gen_ai.provider.name");
    static final AttributeKey<String> REQUEST_MODEL = AttributeKey.stringKey("gen_ai.request.model");
    static final AttributeKey<String> RESPONSE_MODEL = AttributeKey.stringKey("gen_ai.response.model");
    static final AttributeKey<String> RESPONSE_ID = AttributeKey.stringKey("gen_ai.response.id");
    static final AttributeKey<Long> INPUT_TOKENS = AttributeKey.longKey("gen_ai.usage.input_tokens");
    static final AttributeKey<Long> OUTPUT_TOKENS = AttributeKey.longKey("gen_ai.usage.output_tokens");
    static final AttributeKey<List<String>> FINISH_REASONS = AttributeKey.stringArrayKey("gen_ai.response.finish_reasons");
    static final AttributeKey<Double> TEMPERATURE = AttributeKey.doubleKey("gen_ai.request.temperature");
    static final AttributeKey<Double> TOP_P = AttributeKey.doubleKey("gen_ai.request.top_p");
    static final AttributeKey<Long> MAX_TOKENS = AttributeKey.longKey("gen_ai.request.max_tokens");
    static final AttributeKey<String> SERVER_ADDRESS = AttributeKey.stringKey("server.address");
    static final AttributeKey<Long> SERVER_PORT = AttributeKey.longKey("server.port");
    static final AttributeKey<String> ERROR_TYPE = AttributeKey.stringKey("error.type");
    static final AttributeKey<String> INPUT_MESSAGES = AttributeKey.stringKey("gen_ai.input.messages");
    static final AttributeKey<String> OUTPUT_MESSAGES = AttributeKey.stringKey("gen_ai.output.messages");
    static final AttributeKey<String> SYSTEM_INSTRUCTIONS = AttributeKey.stringKey("gen_ai.system_instructions");
    static final AttributeKey<String> CONVERSATION_ID = AttributeKey.stringKey("gen_ai.conversation.id");
    static final AttributeKey<String> OUTPUT_TYPE = AttributeKey.stringKey("gen_ai.output.type");
    static final AttributeKey<Long> EMBEDDING_DIMENSIONS = AttributeKey.longKey("gen_ai.embeddings.dimension.count");

    private static final TextMapGetter<String> TRACE_PARENT = new TextMapGetter<>() {
        @Override
        public Iterable<String> keys(String carrier) {
            return carrier == null ? List.of() : List.of("traceparent");
        }

        @Override
        public String get(String carrier, String key) {
            return carrier != null && "traceparent".equals(key) ? carrier : null;
        }
    };

    private GenAiSpans() {
    }

    static Span start(TelemetrySession session, String operation, String model) {
        var spanName = model == null || model.isBlank() ? operation : operation + " " + model;
        var parent = session.traceParent == null
            ? Context.current()
            : W3CTraceContextPropagator.getInstance().extract(Context.current(), session.traceParent, TRACE_PARENT);

        var span = session.tracer.spanBuilder(spanName)
            .setParent(parent)
            .setSpanKind(SpanKind.CLIENT)
            .startSpan();
        span.setAttribute(OPERATION, operation);
        span.setAttribute(PROVIDER, session.providerName);
        put(span, REQUEST_MODEL, model);
        put(span, SERVER_ADDRESS, session.serverAddress);
        if (session.serverPort != null && session.serverAddress != null) {
            span.setAttribute(SERVER_PORT, session.serverPort);
        }
        put(span, CONVERSATION_ID, session.conversationId);
        return span;
    }

    static void fail(TelemetrySession session, Span span, Throwable error) {
        if (span == null) {
            return;
        }
        try {
            var type = error == null ? "_OTHER" : error.getClass().getSimpleName();
            if (type == null || type.isBlank()) {
                type = "_OTHER";
            }
            span.setAttribute(ERROR_TYPE, type);
            // The exception message is omitted: provider errors often echo the prompt or part of the request.
            span.setStatus(StatusCode.ERROR, type);
        } catch (Exception e) {
            session.logger.debug("Failed to record an LLM span error", e);
        } finally {
            span.end();
        }
    }

    static void end(Span span) {
        if (span != null) {
            span.end();
        }
    }

    static void put(Span span, AttributeKey<String> key, String value) {
        if (value != null && !value.isBlank()) {
            span.setAttribute(key, value);
        }
    }

    static void put(Span span, AttributeKey<Double> key, Double value) {
        if (value != null) {
            span.setAttribute(key, value);
        }
    }

    static void put(Span span, AttributeKey<Long> key, Integer value) {
        if (value != null) {
            span.setAttribute(key, value.longValue());
        }
    }

    static String modelOrFallback(String requested, String fallback) {
        if (requested != null && !requested.isBlank()) {
            return requested;
        }
        return fallback == null || fallback.isBlank() ? null : fallback;
    }
}
