package io.kestra.plugin.ai.completion;

import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.PdfFileContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatMessage;
import jakarta.annotation.Nullable;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.net.URI;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

public final class CompletionInputContentUtils {
    private static final String PDF_MIME_TYPE = "application/pdf";

    private CompletionInputContentUtils() {
    }

    public static UserMessage toUserMessage(RunContext runContext, @Nullable String prompt, @Nullable List<ChatMessage.ContentBlock> promptContentBlocks) throws Exception {
        boolean hasPrompt = prompt != null && !prompt.isBlank();
        if (hasPrompt) {
            return UserMessage.userMessage(prompt);
        }

        if (promptContentBlocks == null || promptContentBlocks.isEmpty()) {
            throw new IllegalArgumentException("At least one input content block must be provided.");
        }

        List<Content> contents = new ArrayList<>(promptContentBlocks.size());
        for (ChatMessage.ContentBlock block : promptContentBlocks) {
            contents.add(switch (block.effectiveType()) {
                case TEXT -> toTextContent(block);
                case IMAGE -> toImageContent(runContext, block);
                case PDF -> toPdfContent(runContext, block);
            });
        }

        if (contents.isEmpty()) {
            throw new IllegalArgumentException("`promptContentBlocks` must contain at least one content block.");
        }

        return UserMessage.userMessage(contents);
    }

    private static TextContent toTextContent(ChatMessage.ContentBlock block) {
        if (block.text() == null || block.text().isBlank()) {
            throw new IllegalArgumentException("TEXT content blocks require a non-empty `text` field.");
        }
        return TextContent.from(block.text());
    }

    private static ImageContent toImageContent(RunContext runContext, ChatMessage.ContentBlock block) throws Exception {
        if (block.uri() == null || block.uri().isBlank()) {
            throw new IllegalArgumentException("IMAGE content blocks require `uri` pointing to a Kestra uploaded file.");
        }

        URI uri = parseUri(block.uri(), "IMAGE");
        byte[] bytes = resolveKestraFileBytes(runContext, uri, "IMAGE");
        String mediaType = resolveImageMediaType(bytes);
        return ImageContent.from(Base64.getEncoder().encodeToString(bytes), mediaType);
    }

    private static PdfFileContent toPdfContent(RunContext runContext, ChatMessage.ContentBlock block) throws Exception {
        if (block.uri() == null || block.uri().isBlank()) {
            throw new IllegalArgumentException("PDF content blocks require `uri` pointing to a Kestra uploaded file.");
        }

        URI uri = parseUri(block.uri(), "PDF");
        byte[] bytes = resolveKestraFileBytes(runContext, uri, "PDF");
        return PdfFileContent.from(Base64.getEncoder().encodeToString(bytes), PDF_MIME_TYPE);
    }

    private static URI parseUri(String raw, String blockType) {
        try {
            return URI.create(raw);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid " + blockType + " uri: `" + raw + "`", e);
        }
    }

    private static byte[] resolveKestraFileBytes(RunContext runContext, URI uri, String blockType) throws Exception {
        if (!"kestra".equalsIgnoreCase(uri.getScheme())) {
            throw new IllegalArgumentException(blockType + " content supports only Kestra uploaded files (`kestra://...`).");
        }

        try (InputStream inputStream = runContext.storage().getFile(uri)) {
            return inputStream.readAllBytes();
        } catch (Exception e) {
            throw new IllegalArgumentException("Unable to read " + blockType + " file from Kestra storage uri `" + uri + "`.", e);
        }
    }

    private static String resolveImageMediaType(byte[] bytes) throws Exception {
        String mediaType;
        try (ByteArrayInputStream stream = new ByteArrayInputStream(bytes)) {
            mediaType = URLConnection.guessContentTypeFromStream(stream);
        }

        if (mediaType == null || mediaType.isBlank()) {
            throw new IllegalArgumentException("Unable to detect IMAGE media type.");
        }
        if (!mediaType.startsWith("image/")) {
            throw new IllegalArgumentException("Invalid IMAGE media type `" + mediaType + "`. It must start with `image/`.");
        }
        return mediaType;
    }
}
