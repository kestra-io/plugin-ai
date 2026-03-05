package io.kestra.plugin.ai.domain;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;

import java.util.List;

@Builder
@Schema(
    title = "Chat Message",
    description = "A chat message payload. Use either `content` for plain text or `contentBlocks` for multimodal content blocks."
)
public record ChatMessage(
    @Schema(title = "Message type")
    ChatMessageType type,
    @Schema(title = "Text content", description = "Plain text message content. Must be set only when `contentBlocks` is not set.")
    String content,
    @Schema(title = "Content blocks", description = "Multimodal message blocks (TEXT, IMAGE, PDF). Must be set only when `content` is not set.")
    List<ContentBlock> contentBlocks
) {
    public ChatMessage {
        boolean hasTextContent = content != null && !content.isBlank();
        boolean hasBlockContent = contentBlocks != null && !contentBlocks.isEmpty();
        if (hasTextContent == hasBlockContent) {
            throw new IllegalArgumentException("Exactly one of `content` or `contentBlocks` must be provided.");
        }
    }

    public List<ContentBlock> effectiveContents() {
        if (contentBlocks != null && !contentBlocks.isEmpty()) {
            return contentBlocks;
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
