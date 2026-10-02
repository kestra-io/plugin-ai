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
    @Schema(
        title = "Message type",
        description = "Role the message plays in the conversation: `SYSTEM`, `USER` or `AI`. There can be at most one `SYSTEM` message, and the last message must be a `USER` message.",
        example = "USER"
    )
    ChatMessageType type,
    @Schema(
        title = "Text content",
        description = "Plain text body of the message. Mutually exclusive with `contentBlocks`: exactly one of the two must be set.",
        example = "{{ inputs.prompt }}"
    )
    String content,
    @Schema(
        title = "Content blocks",
        description = "Multimodal body of the message, as a list of `TEXT`, `IMAGE` or `PDF` blocks. Mutually exclusive with `content`: exactly one of the two must be set.",
        example = "[{type: \"TEXT\", text: \"What is in this image?\"}, {type: \"IMAGE\", uri: \"{{ inputs.photo }}\"}]"
    )
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
        @Schema(
            title = "Block type",
            description = "Kind of payload this block carries: `TEXT`, `IMAGE` or `PDF`. Defaults to `TEXT` when omitted.",
            example = "IMAGE"
        )
        Type type,
        @Schema(
            title = "Text",
            description = "Text payload of the block. Required for `TEXT` blocks and ignored otherwise.",
            example = "What is in this image?"
        )
        String text,
        @Schema(
            title = "URI",
            description = "Location of the file for `IMAGE` and `PDF` blocks, ignored for `TEXT` blocks. Supports the `kestra://`, `file://` and `nsfile://` smart URI schemes.",
            example = "{{ inputs.photo }}"
        )
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
