package io.kestra.plugin.ai.observability;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.service.Result;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.LangfuseObservability;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.ai.domain.TokenUsage;
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
import io.opentelemetry.sdk.resources.ResourceBuilder;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import io.opentelemetry.sdk.trace.export.SpanExporter;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public final class OpenTelemetryLangfuseObservability implements AgentObservability {
    private static final ObjectMapper MAPPER = JacksonMapper.ofJson(false);
    private static final String TRACE_NAME = "AIAgent";
    private static final String INSTRUMENTATION_SCOPE = "io.kestra.plugin.ai.agent.AIAgent";

    private final RunContext runContext;
    private final String taskId;
    private final String providerType;
    private final String modelName;
    private final ChatConfiguration chatConfiguration;
    private final ResolvedConfig config;
    private final SdkTracerProvider tracerProvider;
    private final OpenTelemetrySdk openTelemetrySdk;
    private final Tracer tracer;

    private Span span;

    private OpenTelemetryLangfuseObservability(
        RunContext runContext,
        String taskId,
        String providerType,
        String modelName,
        ChatConfiguration chatConfiguration,
        ResolvedConfig config,
        SpanExporter spanExporter
    ) {
        this.runContext = runContext;
        this.taskId = taskId;
        this.providerType = providerType;
        this.modelName = modelName;
        this.chatConfiguration = chatConfiguration;
        this.config = config;

        ResourceBuilder resourceBuilder = Resource.builder()
            .put("service.name", config.serviceName());

        if (config.environment() != null) {
            resourceBuilder.put("deployment.environment", config.environment());
        }
        if (config.release() != null) {
            resourceBuilder.put("service.version", config.release());
        }

        this.tracerProvider = SdkTracerProvider.builder()
            .setResource(Resource.getDefault().merge(resourceBuilder.build()))
            .addSpanProcessor(BatchSpanProcessor.builder(spanExporter).build())
            .build();

        this.openTelemetrySdk = OpenTelemetrySdk.builder()
            .setTracerProvider(this.tracerProvider)
            .build();

        this.tracer = this.openTelemetrySdk.getTracer(INSTRUMENTATION_SCOPE);
    }

    public static AgentObservability create(
        RunContext runContext,
        LangfuseObservability langfuse,
        String taskId,
        ModelProvider provider,
        ChatConfiguration chatConfiguration
    ) {
        try {
            ResolvedConfig resolvedConfig = ResolvedConfig.from(runContext, langfuse);
            if (!resolvedConfig.enabled()) {
                return NoopAgentObservability.INSTANCE;
            }

            String providerType = provider != null ? provider.getClass().getSimpleName() : null;
            String modelName = null;
            if (provider != null && provider.getModelName() != null) {
                modelName = runContext.render(provider.getModelName()).as(String.class).orElse(null);
            }

            SpanExporter exporter = OtlpHttpSpanExporter.builder()
                .setEndpoint(resolvedConfig.endpoint())
                .addHeader("Authorization", basicAuth(resolvedConfig.publicKey(), resolvedConfig.secretKey()))
                .setTimeout(resolvedConfig.exportTimeout())
                .build();

            return new OpenTelemetryLangfuseObservability(
                runContext,
                taskId,
                providerType,
                modelName,
                chatConfiguration,
                resolvedConfig,
                exporter
            );
        } catch (Exception e) {
            runContext.logger().warn("Failed to initialize Langfuse observability. Falling back to no-op.", e);
            return NoopAgentObservability.INSTANCE;
        }
    }

    @Override
    public void onStart(String prompt, String systemMessage) {
        try {
            span = tracer.spanBuilder("ai.agent.run")
                .setSpanKind(SpanKind.INTERNAL)
                .startSpan();

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

            Map<String, Object> traceMetadata = traceMetadata(systemMessage);
            if (!traceMetadata.isEmpty()) {
                span.setAttribute("langfuse.trace.metadata", toJson(traceMetadata));
            }

            String correlationId = correlationId();
            if (correlationId != null) {
                span.setAttribute("langfuse.session.id", correlationId);
            }

            Map<String, Object> modelParameters = modelParameters();
            if (!modelParameters.isEmpty()) {
                span.setAttribute("langfuse.observation.model.parameters", toJson(modelParameters));
            }

            if (config.capturePrompt()) {
                String sanitizedPrompt = sanitize(prompt);
                span.setAttribute("langfuse.trace.input", sanitizedPrompt);
                span.setAttribute("langfuse.observation.input", sanitizedPrompt);
            }
        } catch (Exception e) {
            runContext.logger().warn("Failed to start Langfuse span", e);
        }
    }

    @Override
    public void onToolArgumentsError(String toolName, String requestId, Throwable error) {
        if (span == null) {
            return;
        }

        AttributesBuilder attributesBuilder = Attributes.builder();
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

    @Override
    public void onToolExecutionError(String toolName, String requestId, Throwable error) {
        if (span == null) {
            return;
        }

        AttributesBuilder attributesBuilder = Attributes.builder();
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

    @Override
    public void onCompletion(Result<AiMessage> completion, TokenUsage tokenUsage) {
        if (span == null) {
            return;
        }

        try {
            if (tokenUsage != null) {
                Map<String, Object> usageDetails = new LinkedHashMap<>();
                putIfNotNull(usageDetails, "input", tokenUsage.getInputTokenCount());
                putIfNotNull(usageDetails, "output", tokenUsage.getOutputTokenCount());
                putIfNotNull(usageDetails, "total", tokenUsage.getTotalTokenCount());
                if (!usageDetails.isEmpty()) {
                    span.setAttribute("langfuse.observation.usage_details", toJson(usageDetails));
                }
            }

            if (completion != null && completion.finishReason() != null) {
                span.setAttribute("langfuse.observation.level", "DEFAULT");
                span.setAttribute("ai.finish.reason", completion.finishReason().name());
            }

            if (config.captureOutput() && completion != null && completion.content() != null) {
                String output = sanitize(completion.content().text());
                if (output != null) {
                    span.setAttribute("langfuse.trace.output", output);
                    span.setAttribute("langfuse.observation.output", output);
                }
            }

            if (completion != null && completion.toolExecutions() != null) {
                completion.toolExecutions().forEach(toolExecution -> {
                    AttributesBuilder attrs = Attributes.builder();

                    ToolExecutionRequest request = toolExecution.request();
                    if (request != null) {
                        putString(attrs, "tool.name", request.name());
                        putString(attrs, "tool.request.id", request.id());
                        if (config.captureToolArguments()) {
                            putString(attrs, "tool.arguments", sanitize(request.arguments()));
                        }
                    }

                    if (config.captureToolResults()) {
                        putString(attrs, "tool.result", sanitize(toolExecution.result()));
                    }

                    span.addEvent("tool.execution", attrs.build());
                });
            }

            span.setStatus(StatusCode.OK);
        } catch (Exception e) {
            runContext.logger().warn("Failed to enrich Langfuse span on completion", e);
        }
    }

    @Override
    public void onFailure(Throwable error) {
        if (span == null) {
            return;
        }

        try {
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

    @Override
    public void close() {
        if (span != null) {
            try {
                span.end();
            } catch (Exception e) {
                runContext.logger().warn("Failed to end Langfuse span", e);
            }
        }

        waitFor("forceFlush", tracerProvider.forceFlush());
        waitFor("shutdown", tracerProvider.shutdown());

        try {
            openTelemetrySdk.close();
        } catch (Exception e) {
            runContext.logger().warn("Failed to close OpenTelemetry SDK", e);
        }
    }

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

    private Map<String, Object> traceMetadata(String systemMessage) {
        Map<String, Object> metadata = new LinkedHashMap<>();

        if (taskId != null) {
            metadata.put("taskId", taskId);
        }

        if (runContext.flowInfo() != null) {
            putIfNotNull(metadata, "flowId", runContext.flowInfo().id());
            putIfNotNull(metadata, "namespace", runContext.flowInfo().namespace());
        }

        String executionId = nestedLabel("labels", "system", "executionId");
        putIfNotNull(metadata, "executionId", executionId);

        String taskRunId = nestedLabel("taskrun", "id");
        putIfNotNull(metadata, "taskRunId", taskRunId);

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

        String sanitized = value.strip();
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
        String credentials = publicKey + ":" + secretKey;
        String encoded = Base64.getEncoder().encodeToString(credentials.getBytes(StandardCharsets.UTF_8));
        return "Basic " + encoded;
    }

    private record ResolvedConfig(
        boolean enabled,
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
        Duration exportTimeout
    ) {
        private static ResolvedConfig from(RunContext runContext, LangfuseObservability raw) throws IllegalVariableEvaluationException {
            if (raw == null) {
                return disabled();
            }

            boolean enabled = render(runContext, raw.getEnabled(), Boolean.class, false);
            if (!enabled) {
                return disabled();
            }

            String endpoint = normalizeEndpoint(render(runContext, raw.getEndpoint(), String.class, null));
            String publicKey = render(runContext, raw.getPublicKey(), String.class, null);
            String secretKey = render(runContext, raw.getSecretKey(), String.class, null);

            if (isBlank(endpoint) || isBlank(publicKey) || isBlank(secretKey)) {
                throw new IllegalArgumentException("Langfuse endpoint, publicKey and secretKey are required when observability is enabled.");
            }

            int maxPayloadChars = render(runContext, raw.getMaxPayloadChars(), Integer.class, 2000);
            if (maxPayloadChars <= 0) {
                maxPayloadChars = 2000;
            }

            Duration timeout = render(runContext, raw.getExportTimeout(), Duration.class, Duration.ofSeconds(5));
            if (timeout == null || timeout.isNegative() || timeout.isZero()) {
                timeout = Duration.ofSeconds(5);
            }

            return new ResolvedConfig(
                true,
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

        private static ResolvedConfig disabled() {
            return new ResolvedConfig(
                false,
                null,
                null,
                null,
                "kestra-plugin-ai",
                null,
                null,
                false,
                false,
                false,
                false,
                false,
                2000,
                Duration.ofSeconds(5)
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

            String normalized = endpoint.strip();
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
