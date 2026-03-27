package io.kestra.plugin.ai.observability;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.DefaultRunContext;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.LangfuseObservability;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.ai.domain.Observability;
import io.kestra.plugin.ai.domain.TokenUsage;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.observability.api.event.AiServiceCompletedEvent;
import dev.langchain4j.observability.api.event.AiServiceErrorEvent;
import dev.langchain4j.observability.api.event.AiServiceStartedEvent;
import dev.langchain4j.observability.api.event.ToolExecutedEvent;
import dev.langchain4j.observability.api.listener.AiServiceCompletedListener;
import dev.langchain4j.observability.api.listener.AiServiceErrorListener;
import dev.langchain4j.observability.api.listener.AiServiceListener;
import dev.langchain4j.observability.api.listener.AiServiceStartedListener;
import dev.langchain4j.observability.api.listener.ToolExecutedEventListener;
import dev.langchain4j.service.Result;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributesBuilder;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.exporter.otlp.http.trace.OtlpHttpSpanExporter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import io.opentelemetry.sdk.trace.export.SpanExporter;

public final class LangfuseObservabilityListeners implements AutoCloseable {
    private static final ObjectMapper MAPPER = JacksonMapper.ofJson(false);
    private static final String TRACE_NAME = "AIAgent";
    private static final String INSTRUMENTATION_SCOPE = "io.kestra.plugin.ai.agent.AIAgent";

    /** No-op instance returned when observability is disabled. */
    private static final LangfuseObservabilityListeners NOOP = new LangfuseObservabilityListeners();

    private final RunContext runContext;
    private final String taskId;
    private final String providerType;
    private final String modelName;
    private final ChatConfiguration chatConfiguration;
    private final ResolvedConfig config;
    private final SdkTracerProvider tracerProvider;
    private final OpenTelemetrySdk openTelemetrySdk;
    private final Tracer tracer;
    private final boolean sharedOpenTelemetry;
    private final boolean enabled;
    private final LangfuseChatModelListener chatModelListener;

    private Span span;

    // NOOP constructor
    private LangfuseObservabilityListeners() {
        this.runContext = null;
        this.taskId = null;
        this.providerType = null;
        this.modelName = null;
        this.chatConfiguration = null;
        this.config = null;
        this.tracerProvider = null;
        this.openTelemetrySdk = null;
        this.tracer = null;
        this.sharedOpenTelemetry = false;
        this.enabled = false;
        this.chatModelListener = null;
    }

    private LangfuseObservabilityListeners(
        RunContext runContext,
        String taskId,
        String providerType,
        String modelName,
        ChatConfiguration chatConfiguration,
        ResolvedConfig config,
        Tracer tracer,
        SdkTracerProvider tracerProvider,
        OpenTelemetrySdk openTelemetrySdk,
        boolean sharedOpenTelemetry) {
        this.runContext = runContext;
        this.taskId = taskId;
        this.providerType = providerType;
        this.modelName = modelName;
        this.chatConfiguration = chatConfiguration;
        this.config = config;
        this.tracer = tracer;
        this.tracerProvider = tracerProvider;
        this.openTelemetrySdk = openTelemetrySdk;
        this.sharedOpenTelemetry = sharedOpenTelemetry;
        this.enabled = true;
        this.chatModelListener = new LangfuseChatModelListener(tracer, runContext.logger());
    }

    // --- Factory methods ---

    private static LangfuseObservabilityListeners isolated(
        RunContext runContext,
        String taskId,
        String providerType,
        String modelName,
        ChatConfiguration chatConfiguration,
        ResolvedConfig config,
        SpanExporter spanExporter) {
        var resourceBuilder = Resource.builder()
            .put("service.name", config.serviceName());

        if (config.environment() != null) {
            resourceBuilder.put("deployment.environment", config.environment());
        }
        if (config.release() != null) {
            resourceBuilder.put("service.version", config.release());
        }

        var sdkTracerProvider = SdkTracerProvider.builder()
            .setResource(Resource.getDefault().merge(resourceBuilder.build()))
            .addSpanProcessor(BatchSpanProcessor.builder(spanExporter).build())
            .build();

        var sdk = OpenTelemetrySdk.builder()
            .setTracerProvider(sdkTracerProvider)
            .build();

        var otelTracer = sdk.getTracer(INSTRUMENTATION_SCOPE);

        return new LangfuseObservabilityListeners(
            runContext, taskId, providerType, modelName, chatConfiguration,
            config, otelTracer, sdkTracerProvider, sdk, false
        );
    }

    private static LangfuseObservabilityListeners shared(
        RunContext runContext,
        String taskId,
        String providerType,
        String modelName,
        ChatConfiguration chatConfiguration,
        ResolvedConfig config,
        OpenTelemetry openTelemetry) {
        var otelTracer = openTelemetry.getTracer(INSTRUMENTATION_SCOPE);
        return new LangfuseObservabilityListeners(
            runContext, taskId, providerType, modelName, chatConfiguration,
            config, otelTracer, null, null, true
        );
    }

    public static LangfuseObservabilityListeners create(
        RunContext runContext,
        Observability observability,
        String taskId,
        ModelProvider provider,
        ChatConfiguration chatConfiguration) {
        try {
            var sharedOtel = resolveOpenTelemetryBean(runContext);
            var resolvedConfig = ResolvedConfig.from(runContext, observability);
            if (resolvedConfig == null) {
                return NOOP;
            }

            var provType = provider != null ? provider.getClass().getSimpleName() : null;
            String mdlName = null;
            if (provider != null && provider.getModelName() != null) {
                mdlName = runContext.render(provider.getModelName()).as(String.class).orElse(null);
            }

            if (sharedOtel != null && !resolvedConfig.hasExporterConfiguration()) {
                runContext.logger().debug("Reusing existing OpenTelemetry bean for Langfuse observability.");
                return shared(runContext, taskId, provType, mdlName, chatConfiguration, resolvedConfig, sharedOtel);
            }

            if (!resolvedConfig.hasExporterConfiguration()) {
                throw new IllegalArgumentException("Langfuse endpoint, publicKey and secretKey are required when observability is enabled.");
            }

            SpanExporter exporter = OtlpHttpSpanExporter.builder()
                .setEndpoint(resolvedConfig.endpoint())
                .addHeader("Authorization", basicAuth(resolvedConfig.publicKey(), resolvedConfig.secretKey()))
                .setTimeout(resolvedConfig.exportTimeout())
                .build();

            return isolated(runContext, taskId, provType, mdlName, chatConfiguration, resolvedConfig, exporter);
        } catch (Exception e) {
            runContext.logger().warn("Failed to initialize Langfuse observability. Falling back to no-op.", e);
            return NOOP;
        }
    }

    // --- Public API for AIAgent/ChatCompletion integration ---

    /** Returns AI Service listeners to register via AiServices.builder().registerListeners(...) */
    public List<AiServiceListener<?>> aiServiceListeners() {
        if (!enabled) {
            return Collections.emptyList();
        }
        return List.of(
            (AiServiceStartedListener) this::handleStarted,
            (AiServiceCompletedListener) this::handleCompleted,
            (AiServiceErrorListener) this::handleError,
            (ToolExecutedEventListener) this::handleToolExecuted
        );
    }

    /** Returns ChatModelListeners to pass to provider.chatModel(..., additionalListeners) */
    public List<ChatModelListener> chatModelListeners() {
        if (!enabled || chatModelListener == null) {
            return Collections.emptyList();
        }
        return List.of(chatModelListener);
    }

    // --- Tool error methods (called directly from AIAgent error handlers, not via LC4j listeners) ---

    public void onToolArgumentsError(String toolName, String requestId, Throwable error) {
        if (!enabled || span == null) {
            return;
        }

        var attributesBuilder = Attributes.builder();
        putString(attributesBuilder, "tool.name", toolName);
        putString(attributesBuilder, "tool.request.id", requestId);
        putString(attributesBuilder, "error.type", error != null ? error.getClass().getSimpleName() : null);
        putString(attributesBuilder, "error.message", sanitize(error != null ? error.getMessage() : null));

        span.addEvent("tool.arguments.error", attributesBuilder.build());
        if (error != null) {
            span.recordException(error);
        }
        span.setStatus(StatusCode.ERROR, "Tool arguments error");
    }

    public void onToolExecutionError(String toolName, String requestId, Throwable error) {
        if (!enabled || span == null) {
            return;
        }

        var attributesBuilder = Attributes.builder();
        putString(attributesBuilder, "tool.name", toolName);
        putString(attributesBuilder, "tool.request.id", requestId);
        putString(attributesBuilder, "error.type", error != null ? error.getClass().getSimpleName() : null);
        putString(attributesBuilder, "error.message", sanitize(error != null ? error.getMessage() : null));

        span.addEvent("tool.execution.error", attributesBuilder.build());
        if (error != null) {
            span.recordException(error);
        }
        span.setStatus(StatusCode.ERROR, "Tool execution error");
    }

    // --- LC4j Listener handlers ---

    private void handleStarted(AiServiceStartedEvent event) {
        try {
            span = tracer.spanBuilder("ai.agent.run")
                .setSpanKind(SpanKind.INTERNAL)
                .startSpan();

            // Set parent span for chat model child spans
            if (chatModelListener != null) {
                chatModelListener.setParentSpan(span);
            }

            span.setAttribute("langfuse.trace.name", TRACE_NAME);
            span.setAttribute("langfuse.observation.type", "generation");

            if (modelName != null) {
                span.setAttribute("langfuse.observation.model", modelName);
                span.setAttribute("gen_ai.request.model", modelName);
            }
            if (providerType != null) {
                span.setAttribute("langfuse.observation.model.provider", providerType);
            }

            List<String> tags = new ArrayList<>();
            tags.add("kestra");
            tags.add("ai-agent");
            if (providerType != null) {
                tags.add(providerType.toLowerCase());
            }
            span.setAttribute(AttributeKey.stringArrayKey("langfuse.trace.tags"), tags);

            // Extract system message from event
            String systemMessage = event.systemMessage().map(sm -> sm.text()).orElse(null);

            Map<String, Object> traceMetadata = traceMetadata(systemMessage);
            if (!traceMetadata.isEmpty()) {
                span.setAttribute("langfuse.trace.metadata", toJson(traceMetadata));
            }

            String corrId = correlationId();
            if (corrId != null) {
                span.setAttribute("langfuse.session.id", corrId);
            }

            Map<String, Object> params = modelParameters();
            if (!params.isEmpty()) {
                span.setAttribute("langfuse.observation.model.parameters", toJson(params));
            }

            if (config.capturePrompt()) {
                var userMessage = event.userMessage();
                if (userMessage != null && userMessage.hasSingleText()) {
                    var sanitizedPrompt = sanitize(userMessage.singleText());
                    span.setAttribute("langfuse.trace.input", sanitizedPrompt);
                    span.setAttribute("langfuse.observation.input", sanitizedPrompt);
                }
            }
        } catch (Exception e) {
            runContext.logger().warn("Failed to start Langfuse span", e);
        }
    }

    private void handleToolExecuted(ToolExecutedEvent event) {
        if (span == null) {
            return;
        }

        try {
            var attrs = Attributes.builder();
            var request = event.request();
            if (request != null) {
                putString(attrs, "tool.name", request.name());
                putString(attrs, "tool.request.id", request.id());
                if (config.captureToolArguments()) {
                    putString(attrs, "tool.arguments", sanitize(request.arguments()));
                }
            }
            if (config.captureToolResults()) {
                putString(attrs, "tool.result", sanitize(event.resultText()));
            }
            span.addEvent("tool.execution", attrs.build());
        } catch (Exception e) {
            runContext.logger().debug("Failed to add tool execution event to Langfuse span", e);
        }
    }

    @SuppressWarnings("unchecked")
    private void handleCompleted(AiServiceCompletedEvent event) {
        if (span == null) {
            return;
        }

        try {
            // Try to extract Result<AiMessage> from the event result
            var result = event.result().orElse(null);
            if (result instanceof Result<?> typedResult) {
                // Extract token usage
                if (typedResult.tokenUsage() != null) {
                    var tokenUsage = TokenUsage.from(typedResult.tokenUsage());
                    Map<String, Object> usageDetails = new LinkedHashMap<>();
                    putIfNotNull(usageDetails, "input", tokenUsage.getInputTokenCount());
                    putIfNotNull(usageDetails, "output", tokenUsage.getOutputTokenCount());
                    putIfNotNull(usageDetails, "total", tokenUsage.getTotalTokenCount());
                    if (!usageDetails.isEmpty()) {
                        span.setAttribute("langfuse.observation.usage_details", toJson(usageDetails));
                    }
                }

                if (typedResult.finishReason() != null) {
                    span.setAttribute("langfuse.observation.level", "DEFAULT");
                    span.setAttribute("ai.finish.reason", typedResult.finishReason().name());
                }

                if (config.captureOutput() && typedResult.content() instanceof AiMessage aiMsg) {
                    var output = sanitize(aiMsg.text());
                    if (output != null) {
                        span.setAttribute("langfuse.trace.output", output);
                        span.setAttribute("langfuse.observation.output", output);
                    }
                }
            }

            span.setStatus(StatusCode.OK);
        } catch (Exception e) {
            runContext.logger().warn("Failed to enrich Langfuse span on completion", e);
        }
    }

    private void handleError(AiServiceErrorEvent event) {
        if (span == null) {
            return;
        }

        try {
            var error = event.error();
            if (error != null) {
                span.recordException(error);
                span.setStatus(StatusCode.ERROR, sanitize(error.getMessage()));
            } else {
                span.setStatus(StatusCode.ERROR);
            }
        } catch (Exception e) {
            runContext.logger().warn("Failed to mark Langfuse span as failed", e);
        }
    }

    // --- close() ---

    @Override
    public void close() {
        if (!enabled) {
            return;
        }

        if (span != null) {
            try {
                span.end();
            } catch (Exception e) {
                runContext.logger().warn("Failed to end Langfuse span", e);
            }
        }

        if (sharedOpenTelemetry) {
            return;
        }

        if (tracerProvider == null || openTelemetrySdk == null) {
            return;
        }

        waitFor("forceFlush", tracerProvider.forceFlush());
        waitFor("shutdown", tracerProvider.shutdown());

        try {
            openTelemetrySdk.close();
        } catch (Exception e) {
            runContext.logger().warn("Failed to close OpenTelemetry SDK", e);
        }
    }

    // --- Private helpers (preserved from OpenTelemetryLangfuseObservability) ---

    private void waitFor(String operation, CompletableResultCode resultCode) {
        try {
            resultCode.join(config.exportTimeout().toMillis(), TimeUnit.MILLISECONDS);
            if (!resultCode.isDone()) {
                runContext.logger().warn("OpenTelemetry {} timed out after {} ms", operation, config.exportTimeout().toMillis());
            }
        } catch (Exception e) {
            runContext.logger().warn("OpenTelemetry {} failed", operation, e);
        }
    }

    @SuppressWarnings("deprecation")
    private static OpenTelemetry resolveOpenTelemetryBean(RunContext runContext) {
        if (!(runContext instanceof DefaultRunContext defaultRunContext)) {
            return null;
        }

        try {
            return defaultRunContext.getApplicationContext().findBean(OpenTelemetry.class).orElse(null);
        } catch (Exception e) {
            runContext.logger().debug("Unable to resolve OpenTelemetry bean from run context", e);
            return null;
        }
    }

    private Map<String, Object> traceMetadata(String systemMessage) {
        Map<String, Object> metadata = new LinkedHashMap<>();

        if (taskId != null) {
            metadata.put("taskId", taskId);
        }

        if (runContext.flowInfo() != null) {
            putIfNotNull(metadata, "flowId", runContext.flowInfo().id());
            putIfNotNull(metadata, "namespace", runContext.flowInfo().namespace());
        }

        var executionId = nestedLabel("labels", "system", "executionId");
        putIfNotNull(metadata, "executionId", executionId);

        var taskRunId = nestedLabel("taskrun", "id");
        putIfNotNull(metadata, "taskRunId", taskRunId);

        putIfNotNull(metadata, "parentTraceId", runContext.getTraceParent());

        if (config.captureSystemMessage() && systemMessage != null) {
            metadata.put("systemMessage", sanitize(systemMessage));
        }

        return metadata;
    }

    private Map<String, Object> modelParameters() {
        Map<String, Object> parameters = new LinkedHashMap<>();

        try {
            putIfNotNull(parameters, "temperature", render(chatConfiguration.getTemperature(), Double.class));
            putIfNotNull(parameters, "topP", render(chatConfiguration.getTopP(), Double.class));
            putIfNotNull(parameters, "topK", render(chatConfiguration.getTopK(), Integer.class));
            putIfNotNull(parameters, "seed", render(chatConfiguration.getSeed(), Integer.class));
            putIfNotNull(parameters, "maxToken", render(chatConfiguration.getMaxToken(), Integer.class));
            putIfNotNull(parameters, "returnThinking", render(chatConfiguration.getReturnThinking(), Boolean.class));
        } catch (Exception e) {
            runContext.logger().debug("Unable to resolve full model parameters for Langfuse metadata", e);
        }

        return parameters;
    }

    private String correlationId() {
        return nestedLabel("labels", "system", "correlationId");
    }

    private String nestedLabel(String... path) {
        Object current = runContext.getVariables();
        for (String key : path) {
            if (!(current instanceof Map<?, ?> map)) {
                return null;
            }
            current = map.get(key);
            if (current == null) {
                return null;
            }
        }
        return String.valueOf(current);
    }

    private String sanitize(String value) {
        if (value == null) {
            return null;
        }

        var sanitized = value.strip();
        if (sanitized.length() > config.maxPayloadChars()) {
            return sanitized.substring(0, config.maxPayloadChars()) + "...[truncated]";
        }
        return sanitized;
    }

    private <T> T render(Property<T> property, Class<T> clazz) throws IllegalVariableEvaluationException {
        return runContext.render(property).as(clazz).orElse(null);
    }

    private static void putString(AttributesBuilder attributesBuilder, String key, String value) {
        if (value != null) {
            attributesBuilder.put(key, value);
        }
    }

    private static void putIfNotNull(Map<String, Object> map, String key, Object value) {
        if (value != null) {
            map.put(key, value);
        }
    }

    private String toJson(Object value) {
        if (value == null) {
            return null;
        }

        try {
            return MAPPER.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            return sanitize(String.valueOf(value));
        }
    }

    private static String basicAuth(String publicKey, String secretKey) {
        var credentials = publicKey + ":" + secretKey;
        var encoded = Base64.getEncoder().encodeToString(credentials.getBytes(StandardCharsets.UTF_8));
        return "Basic " + encoded;
    }

    private record ResolvedConfig(
        String endpoint,
        String publicKey,
        String secretKey,
        String serviceName,
        String environment,
        String release,
        boolean capturePrompt,
        boolean captureSystemMessage,
        boolean captureOutput,
        boolean captureToolArguments,
        boolean captureToolResults,
        int maxPayloadChars,
        Duration exportTimeout) {
        private boolean hasExporterConfiguration() {
            return !isBlank(endpoint) && !isBlank(publicKey) && !isBlank(secretKey);
        }

        private static ResolvedConfig from(RunContext runContext, Observability raw) throws IllegalVariableEvaluationException {
            if (raw == null) {
                return null;
            }

            // Langfuse-specific fields: only available when the provider is LangfuseObservability
            String endpoint = null;
            String publicKey = null;
            String secretKey = null;
            if (raw instanceof LangfuseObservability langfuse) {
                endpoint = normalizeEndpoint(render(runContext, langfuse.getEndpoint(), String.class, null));
                publicKey = render(runContext, langfuse.getPublicKey(), String.class, null);
                secretKey = render(runContext, langfuse.getSecretKey(), String.class, null);
            }

            int maxPayloadChars = render(runContext, raw.getMaxPayloadChars(), Integer.class, 2000);
            if (maxPayloadChars <= 0) {
                maxPayloadChars = 2000;
            }

            var timeout = render(runContext, raw.getExportTimeout(), Duration.class, Duration.ofSeconds(5));
            if (timeout == null || timeout.isNegative() || timeout.isZero()) {
                timeout = Duration.ofSeconds(5);
            }

            return new ResolvedConfig(
                endpoint,
                publicKey,
                secretKey,
                render(runContext, raw.getServiceName(), String.class, "kestra-plugin-ai"),
                render(runContext, raw.getEnvironment(), String.class, null),
                render(runContext, raw.getRelease(), String.class, null),
                render(runContext, raw.getCapturePrompt(), Boolean.class, false),
                render(runContext, raw.getCaptureSystemMessage(), Boolean.class, false),
                render(runContext, raw.getCaptureOutput(), Boolean.class, false),
                render(runContext, raw.getCaptureToolArguments(), Boolean.class, false),
                render(runContext, raw.getCaptureToolResults(), Boolean.class, false),
                maxPayloadChars,
                timeout
            );
        }

        private static <T> T render(RunContext runContext, Property<T> property, Class<T> clazz, T defaultValue) throws IllegalVariableEvaluationException {
            if (property == null) {
                return defaultValue;
            }
            return runContext.render(property).as(clazz).orElse(defaultValue);
        }

        private static String normalizeEndpoint(String endpoint) {
            if (isBlank(endpoint)) {
                return endpoint;
            }

            var normalized = endpoint.strip();
            if (normalized.endsWith("/")) {
                normalized = normalized.substring(0, normalized.length() - 1);
            }
            if (!normalized.endsWith("/v1/traces") && normalized.contains("/api/public/otel")) {
                normalized = normalized + "/v1/traces";
            }
            return normalized;
        }

        private static boolean isBlank(String value) {
            return value == null || value.isBlank();
        }
    }
}
