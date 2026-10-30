package io.kestra.plugin.ai.observability;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * OTLP endpoint kotlp injects into a wrapped process, read from the standard OpenTelemetry environment variables.
 * HTTP auto-instrumentation does not see LangChain4j's JDK {@code HttpClient} unless a Java agent is attached, and
 * kotlp only sets these variables, so model spans are exported explicitly to this endpoint.
 */
final class OtlpExportConfig {
    static final String CAPTURE_MESSAGE_CONTENT = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";

    private final String tracesEndpoint;
    private final String serviceName;
    private final Map<String, String> headers;
    private final boolean captureMessageContent;

    private OtlpExportConfig(String tracesEndpoint, String serviceName, Map<String, String> headers, boolean captureMessageContent) {
        this.tracesEndpoint = tracesEndpoint;
        this.serviceName = serviceName;
        this.headers = headers;
        this.captureMessageContent = captureMessageContent;
    }

    static OtlpExportConfig fromEnvironment(Map<String, String> env) {
        if (env == null || truthy(env.get("OTEL_SDK_DISABLED"))) {
            return null;
        }

        var tracesExporter = env.get("OTEL_TRACES_EXPORTER");
        if (tracesExporter != null && "none".equalsIgnoreCase(tracesExporter.strip())) {
            return null;
        }

        var tracesEndpoint = tracesEndpoint(env);
        if (tracesEndpoint == null) {
            return null;
        }

        var serviceName = firstNonBlank(env.get("OTEL_SERVICE_NAME"), "kestra-plugin-ai");
        return new OtlpExportConfig(tracesEndpoint, serviceName, headers(env.get("OTEL_EXPORTER_OTLP_HEADERS")), captureMessageContent(env));
    }

    String tracesEndpoint() {
        return tracesEndpoint;
    }

    String serviceName() {
        return serviceName;
    }

    Map<String, String> headers() {
        return headers;
    }

    boolean captureMessageContent() {
        return captureMessageContent;
    }

    static boolean captureMessageContent(Map<String, String> env) {
        var value = env == null ? null : env.get(CAPTURE_MESSAGE_CONTENT);
        if (value == null) {
            return false;
        }
        return switch (value.strip().toLowerCase(Locale.ROOT)) {
            case "true", "1", "yes", "on" -> true;
            default -> false;
        };
    }

    static String tracesEndpoint(Map<String, String> env) {
        var signal = blankToNull(env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"));
        if (signal != null) {
            return stripTrailingSlash(signal);
        }

        var base = blankToNull(env.get("OTEL_EXPORTER_OTLP_ENDPOINT"));
        if (base == null) {
            return null;
        }

        var normalized = stripTrailingSlash(base);
        if (normalized.endsWith("/v1/traces")) {
            return normalized;
        }
        return normalized + "/v1/traces";
    }

    static Map<String, String> headers(String raw) {
        if (raw == null || raw.isBlank()) {
            return Map.of();
        }

        var parsed = new LinkedHashMap<String, String>();
        for (var part : raw.split(",")) {
            var separator = part.indexOf('=');
            if (separator <= 0) {
                continue;
            }
            var key = part.substring(0, separator).strip();
            var value = part.substring(separator + 1).strip();
            if (!key.isEmpty()) {
                parsed.put(key, value);
            }
        }
        return Map.copyOf(parsed);
    }

    private static boolean truthy(String value) {
        if (value == null) {
            return false;
        }
        return switch (value.strip().toLowerCase(Locale.ROOT)) {
            case "true", "1", "yes", "on" -> true;
            default -> false;
        };
    }

    private static String firstNonBlank(String value, String fallback) {
        var present = blankToNull(value);
        return present == null ? fallback : present;
    }

    private static String blankToNull(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return value.strip();
    }

    private static String stripTrailingSlash(String value) {
        var stripped = value.strip();
        while (stripped.endsWith("/")) {
            stripped = stripped.substring(0, stripped.length() - 1);
        }
        return stripped;
    }
}
