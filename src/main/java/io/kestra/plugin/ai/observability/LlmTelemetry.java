package io.kestra.plugin.ai.observability;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;

import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ModelProvider;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.exporter.otlp.http.trace.OtlpHttpSpanExporter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;

/**
 * Exports one GenAI client span per model call when kotlp (or any other OTLP collector) has published
 * {@code OTEL_EXPORTER_OTLP_ENDPOINT}. Disabled otherwise, so tasks that are not being monitored keep the provider
 * model unchanged. A recording OpenTelemetry bean in the worker is not used: wrapping on that bean alone hides
 * provider fields and changes models that tests and callers inspect directly.
 */
public final class LlmTelemetry {
    static final String INSTRUMENTATION_SCOPE = "io.kestra.plugin.ai";

    private static final AttributeKey<String> SERVICE_NAME = AttributeKey.stringKey("service.name");
    private static final Map<String, OpenTelemetrySdk> EXPORTERS = new ConcurrentHashMap<>();
    private static final ThreadLocal<Map<String, String>> ENVIRONMENT = new ThreadLocal<>();
    private static volatile boolean shutdownHookRegistered;

    private LlmTelemetry() {
    }

    public static ChatModel observeChat(RunContext runContext, ModelProvider provider, ChatModel model) {
        if (model == null || model instanceof ObservedChatModel) {
            return model;
        }
        var session = session(runContext, provider);
        return session == null ? model : new ObservedChatModel(model, session);
    }

    public static EmbeddingModel observeEmbedding(RunContext runContext, ModelProvider provider, EmbeddingModel model) {
        if (model == null || model instanceof ObservedEmbeddingModel) {
            return model;
        }
        var session = session(runContext, provider);
        return session == null ? model : new ObservedEmbeddingModel(model, session);
    }

    public static ImageModel observeImage(RunContext runContext, ModelProvider provider, ImageModel model) {
        if (model == null || model instanceof ObservedImageModel) {
            return model;
        }
        var session = session(runContext, provider);
        return session == null ? model : new ObservedImageModel(model, session);
    }

    static TelemetrySession session(RunContext runContext, ModelProvider provider) {
        try {
            var openTelemetry = resolve(runContext);
            if (openTelemetry == null) {
                return null;
            }

            var tracer = openTelemetry.getTracer(INSTRUMENTATION_SCOPE);
            var server = TelemetrySession.serverAddress(render(runContext, provider.telemetryServerUrl()));
            return new TelemetrySession(
                tracer,
                runContext.logger(),
                GenAiProviderNames.fromClass(provider.getClass()),
                render(runContext, provider.getModelName()),
                server == null ? null : server.host(),
                server == null ? null : server.port(),
                OtlpExportConfig.captureMessageContent(environment()),
                runContext.getTraceParent(),
                TelemetrySession.conversationId(runContext.getVariables())
            );
        } catch (Exception e) {
            runContext.logger().debug("LLM OpenTelemetry export is disabled for this task", e);
            return null;
        }
    }

    static OpenTelemetry resolve(RunContext runContext) {
        var config = OtlpExportConfig.fromEnvironment(environment());
        if (config == null) {
            return null;
        }
        return exporter(config, runContext.logger());
    }

    static Map<String, String> environment() {
        var override = ENVIRONMENT.get();
        return override != null ? override : System.getenv();
    }

    static void useEnvironment(Map<String, String> environment) {
        ENVIRONMENT.set(environment);
    }

    static void clearEnvironment() {
        ENVIRONMENT.remove();
    }

    static OpenTelemetrySdk exporter(OtlpExportConfig config, Logger logger) {
        return EXPORTERS.computeIfAbsent(config.tracesEndpoint(), endpoint ->
        {
            var spanExporter = OtlpHttpSpanExporter.builder()
                .setEndpoint(endpoint)
                .setTimeout(Duration.ofSeconds(2))
                .setCompression("none")
                .setHeaders(() -> config.headers())
                .build();

            var provider = SdkTracerProvider.builder()
                .setResource(Resource.getDefault().merge(Resource.builder().put(SERVICE_NAME, config.serviceName()).build()))
                .addSpanProcessor(BatchSpanProcessor.builder(spanExporter).setScheduleDelay(Duration.ofMillis(200)).build())
                .build();

            registerShutdownHook(logger);
            logger.debug("Exporting LLM spans to {}", endpoint);
            return OpenTelemetrySdk.builder().setTracerProvider(provider).build();
        });
    }

    static void flush(String tracesEndpoint) {
        var sdk = tracesEndpoint == null ? null : EXPORTERS.get(tracesEndpoint);
        if (sdk == null) {
            return;
        }
        sdk.getSdkTracerProvider().forceFlush().join(5, TimeUnit.SECONDS);
    }

    static void close(String tracesEndpoint) {
        var sdk = tracesEndpoint == null ? null : EXPORTERS.remove(tracesEndpoint);
        if (sdk != null) {
            sdk.shutdown().join(5, TimeUnit.SECONDS);
        }
    }

    private static String render(RunContext runContext, Property<String> property) {
        if (property == null) {
            return null;
        }
        try {
            return runContext.render(property).as(String.class).orElse(null);
        } catch (Exception e) {
            runContext.logger().debug("Unable to render a value for LLM span attributes", e);
            return null;
        }
    }

    private static void registerShutdownHook(Logger logger) {
        if (shutdownHookRegistered) {
            return;
        }
        synchronized (LlmTelemetry.class) {
            if (shutdownHookRegistered) {
                return;
            }
            Runtime.getRuntime().addShutdownHook(new Thread(() ->
            {
                for (var sdk : EXPORTERS.values()) {
                    try {
                        sdk.shutdown().join(2, TimeUnit.SECONDS);
                    } catch (Exception e) {
                        logger.debug("Failed to flush LLM spans during shutdown", e);
                    }
                }
            }, "kestra-plugin-ai-otel-shutdown"));
            shutdownHookRegistered = true;
        }
    }

    static Tracer tracer(OpenTelemetry openTelemetry) {
        return openTelemetry.getTracer(INSTRUMENTATION_SCOPE);
    }
}
