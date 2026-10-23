package io.kestra.plugin.ai.observability;

import java.net.URI;
import java.util.Map;

import org.slf4j.Logger;

import io.opentelemetry.api.trace.Tracer;

/**
 * Resolved tracer and GenAI attributes shared by every model call in one task.
 */
final class TelemetrySession {
    final Tracer tracer;
    final Logger logger;
    final String providerName;
    final String fallbackModel;
    final String serverAddress;
    final Long serverPort;
    final boolean captureMessageContent;
    final String traceParent;
    final String conversationId;

    TelemetrySession(
        Tracer tracer,
        Logger logger,
        String providerName,
        String fallbackModel,
        String serverAddress,
        Long serverPort,
        boolean captureMessageContent,
        String traceParent,
        String conversationId) {
        this.tracer = tracer;
        this.logger = logger;
        this.providerName = providerName;
        this.fallbackModel = fallbackModel;
        this.serverAddress = serverAddress;
        this.serverPort = serverPort;
        this.captureMessageContent = captureMessageContent;
        this.traceParent = traceParent;
        this.conversationId = conversationId;
    }

    static ServerAddress serverAddress(String rawUrl) {
        if (rawUrl == null || rawUrl.isBlank()) {
            return null;
        }

        var value = rawUrl.strip();
        if (!value.contains("://")) {
            value = "https://" + value;
        }

        try {
            var uri = URI.create(value);
            if (uri.getHost() == null || uri.getHost().isBlank()) {
                return null;
            }
            int port = uri.getPort();
            if (port < 0) {
                port = "http".equalsIgnoreCase(uri.getScheme()) ? 80 : 443;
            }
            return new ServerAddress(uri.getHost(), (long) port);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    static String conversationId(Map<String, Object> variables) {
        if (variables == null) {
            return null;
        }
        var labels = variables.get("labels");
        if (!(labels instanceof Map<?, ?> labelMap)) {
            return null;
        }
        var system = labelMap.get("system");
        if (!(system instanceof Map<?, ?> systemMap)) {
            return null;
        }
        var correlationId = systemMap.get("correlationId");
        if (correlationId == null) {
            return null;
        }
        var rendered = String.valueOf(correlationId).strip();
        return rendered.isEmpty() ? null : rendered;
    }

    record ServerAddress(String host, long port) {
    }
}
