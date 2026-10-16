package io.kestra.plugin.ai.observability;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.kestra.core.serializers.JacksonMapper;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.output.FinishReason;

/**
 * Serializes prompts and completions into the GenAI semantic-convention JSON shape. Content is opt-in because prompts
 * and responses can carry secrets or personal data, and each text part is capped so a single span cannot grow without
 * bound.
 */
final class GenAiMessages {
    static final int MAX_TEXT_CHARS = 4_096;
    static final int MAX_MESSAGES = 40;

    private static final ObjectMapper MAPPER = JacksonMapper.ofJson(false);

    private GenAiMessages() {
    }

    static String systemInstructions(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty()) {
            return null;
        }

        var parts = new ArrayList<Map<String, Object>>();
        for (var message : messages) {
            if (message instanceof SystemMessage system && system.text() != null && !system.text().isBlank()) {
                parts.add(textPart(system.text()));
            }
        }
        return parts.isEmpty() ? null : toJson(parts);
    }

    static String inputMessages(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty()) {
            return null;
        }

        var encoded = new ArrayList<Map<String, Object>>();
        for (var message : messages) {
            if (message instanceof SystemMessage) {
                continue;
            }
            if (encoded.size() >= MAX_MESSAGES) {
                break;
            }
            var rendered = message(message, null);
            if (rendered != null) {
                encoded.add(rendered);
            }
        }
        return encoded.isEmpty() ? null : toJson(encoded);
    }

    static String outputMessage(AiMessage message, FinishReason finishReason) {
        var rendered = message(message, finishReason);
        return rendered == null ? null : toJson(List.of(rendered));
    }

    static String textMessages(String role, String text) {
        if (text == null || text.isBlank()) {
            return null;
        }
        return toJson(List.of(Map.of("role", role, "parts", List.of(textPart(text)))));
    }

    static String finishReason(FinishReason finishReason) {
        if (finishReason == null) {
            return null;
        }
        return switch (finishReason) {
            case STOP -> "stop";
            case LENGTH -> "length";
            case TOOL_EXECUTION -> "tool_calls";
            case CONTENT_FILTER -> "content_filter";
            case OTHER -> "other";
        };
    }

    private static Map<String, Object> message(ChatMessage message, FinishReason finishReason) {
        if (message instanceof UserMessage user) {
            return roleMessage("user", parts(user.contents()), null);
        }
        if (message instanceof AiMessage ai) {
            var parts = new ArrayList<Map<String, Object>>();
            if (ai.text() != null && !ai.text().isBlank()) {
                parts.add(textPart(ai.text()));
            }
            if (ai.thinking() != null && !ai.thinking().isBlank()) {
                var reasoning = new LinkedHashMap<String, Object>();
                reasoning.put("type", "reasoning");
                reasoning.put("content", truncate(ai.thinking()));
                parts.add(reasoning);
            }
            if (ai.hasToolExecutionRequests()) {
                for (var toolCall : ai.toolExecutionRequests()) {
                    parts.add(toolCall(toolCall));
                }
            }
            return roleMessage("assistant", parts, finishReason);
        }
        if (message instanceof ToolExecutionResultMessage tool) {
            var part = new LinkedHashMap<String, Object>();
            part.put("type", "tool_call_response");
            put(part, "id", tool.id());
            put(part, "name", tool.toolName());
            if (tool.text() != null) {
                part.put("response", truncate(tool.text()));
            }
            if (Boolean.TRUE.equals(tool.isError())) {
                part.put("is_error", true);
            }
            return roleMessage("tool", List.of(part), null);
        }
        return null;
    }

    private static Map<String, Object> roleMessage(String role, List<Map<String, Object>> parts, FinishReason finishReason) {
        if (parts == null || parts.isEmpty()) {
            return null;
        }
        var message = new LinkedHashMap<String, Object>();
        message.put("role", role);
        message.put("parts", parts);
        var reason = finishReason(finishReason);
        if (reason != null) {
            message.put("finish_reason", reason);
        }
        return message;
    }

    private static List<Map<String, Object>> parts(List<Content> contents) {
        if (contents == null || contents.isEmpty()) {
            return List.of();
        }
        var parts = new ArrayList<Map<String, Object>>();
        for (var content : contents) {
            if (content instanceof TextContent text) {
                if (text.text() != null && !text.text().isBlank()) {
                    parts.add(textPart(text.text()));
                }
            } else if (content instanceof ImageContent image) {
                var part = new LinkedHashMap<String, Object>();
                part.put("type", "image");
                var url = image.image() == null || image.image().url() == null ? null : image.image().url().toString();
                if (url != null && !url.startsWith("data:")) {
                    part.put("url", url);
                }
                parts.add(part);
            } else if (content != null && content.type() != null) {
                parts.add(Map.of("type", content.type().toString().toLowerCase()));
            }
        }
        return parts;
    }

    private static Map<String, Object> toolCall(ToolExecutionRequest toolCall) {
        var part = new LinkedHashMap<String, Object>();
        part.put("type", "tool_call");
        put(part, "id", toolCall.id());
        put(part, "name", toolCall.name());
        if (toolCall.arguments() != null && !toolCall.arguments().isBlank()) {
            part.put("arguments", arguments(toolCall.arguments()));
        }
        return part;
    }

    private static Object arguments(String raw) {
        var truncated = truncate(raw);
        try {
            return MAPPER.readTree(truncated);
        } catch (JsonProcessingException e) {
            return truncated;
        }
    }

    private static Map<String, Object> textPart(String text) {
        return Map.of("type", "text", "content", truncate(text));
    }

    private static String truncate(String value) {
        if (value == null || value.length() <= MAX_TEXT_CHARS) {
            return value;
        }
        return value.substring(0, MAX_TEXT_CHARS) + "...[truncated]";
    }

    private static void put(Map<String, Object> target, String key, String value) {
        if (value != null && !value.isBlank()) {
            target.put(key, value);
        }
    }

    private static String toJson(Object value) {
        try {
            return MAPPER.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            return null;
        }
    }
}
