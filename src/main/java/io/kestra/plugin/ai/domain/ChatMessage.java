package io.kestra.plugin.ai.domain;

import lombok.Builder;

@Builder
public record ChatMessage(ChatMessageType type, String content) {
}