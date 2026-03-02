package io.kestra.plugin.ai.domain;

import lombok.Builder;

import java.util.List;

@Builder
public record ChatMessage(
    ChatMessageType type,
    String content,
    List<ContentBlock> contents
) {
    public List<ContentBlock> effectiveContents() {
        if (contents != null && !contents.isEmpty()) {
            return contents;
        }

        if (content != null && !content.isBlank()) {
            return List.of(
                ContentBlock.builder()
                    .type(ContentBlock.Type.TEXT)
                    .text(content)
                    .build()
            );
        }

        return List.of();
    }

    @Builder
    public record ContentBlock(
        Type type,
        String text,
        String uri
    ) {
        public Type effectiveType() {
            return type == null ? Type.TEXT : type;
        }

        public enum Type {
            TEXT,
            IMAGE,
            PDF
        }
    }
}
