package io.kestra.plugin.ai.completion;

import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.PdfFileContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import io.kestra.core.models.property.URIFetcher;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatMessage;
import jakarta.annotation.Nullable;
import org.apache.tika.Tika;

import java.io.InputStream;
import java.net.URI;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

public final class CompletionInputContentUtils {
    private static final String PDF_MIME_TYPE = "application/pdf";
    private static final String OCTET_STREAM_MIME_TYPE = "application/octet-stream";
    private static final Tika TIKA = new Tika();

    private CompletionInputContentUtils() {
    }

    public static UserMessage toUserMessage(RunContext runContext, @Nullable String prompt, @Nullable List<ChatMessage.ContentBlock> contentBlocks) throws Exception {
        validatePromptInput("Input", prompt, contentBlocks);
        if (prompt != null && !prompt.isBlank()) {
            return UserMessage.userMessage(prompt);
        }

        List<Content> contents = new ArrayList<>(contentBlocks.size());
        for (ChatMessage.ContentBlock block : contentBlocks) {
            contents.add(switch (block.effectiveType()) {
                case TEXT -> toTextContent(block);
                case IMAGE -> toImageContent(runContext, block);
                case PDF -> toPdfContent(runContext, block);
            });
        }

        return UserMessage.userMessage(contents);
    }

    public static void validatePromptInput(String taskName, @Nullable String prompt, @Nullable List<ChatMessage.ContentBlock> contentBlocks) {
        boolean hasPrompt = prompt != null && !prompt.isBlank();
        boolean hasContentBlocks = contentBlocks != null && !contentBlocks.isEmpty();
        if (hasPrompt && hasContentBlocks) {
            throw new IllegalArgumentException(taskName + " accepts either `prompt` or `contentBlocks`, but not both.");
        }
        if (!hasPrompt && !hasContentBlocks) {
            throw new IllegalArgumentException(taskName + " requires one input source: `prompt` or `contentBlocks`.");
        }
    }

    private static TextContent toTextContent(ChatMessage.ContentBlock block) {
        if (block.text() == null || block.text().isBlank()) {
            throw new IllegalArgumentException("TEXT content blocks require a non-empty `text` field.");
        }
        return TextContent.from(block.text());
    }

    private static ImageContent toImageContent(RunContext runContext, ChatMessage.ContentBlock block) throws Exception {
        if (block.uri() == null || block.uri().isBlank()) {
            throw new IllegalArgumentException("IMAGE content blocks require `uri` pointing to a supported smart URI (`kestra://`, `file://`, or `nsfile://`).");
        }

        URI uri = parseUri(block.uri(), "IMAGE");
        byte[] bytes = resolveUriBytes(runContext, uri, "IMAGE");
        String mediaType = resolveImageMediaType(bytes);
        return ImageContent.from(Base64.getEncoder().encodeToString(bytes), mediaType);
    }

    private static PdfFileContent toPdfContent(RunContext runContext, ChatMessage.ContentBlock block) throws Exception {
        if (block.uri() == null || block.uri().isBlank()) {
            throw new IllegalArgumentException("PDF content blocks require `uri` pointing to a supported smart URI (`kestra://`, `file://`, or `nsfile://`).");
        }

        URI uri = parseUri(block.uri(), "PDF");
        byte[] bytes = resolveUriBytes(runContext, uri, "PDF");
        return PdfFileContent.from(Base64.getEncoder().encodeToString(bytes), PDF_MIME_TYPE);
    }

    private static URI parseUri(String raw, String blockType) {
        try {
            return URI.create(raw);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid " + blockType + " uri: `" + raw + "`", e);
        }
    }

    private static byte[] resolveUriBytes(RunContext runContext, URI uri, String blockType) throws Exception {
        try (InputStream inputStream = URIFetcher.of(uri).fetch(runContext)) {
            return inputStream.readAllBytes();
        } catch (Exception e) {
            throw new IllegalArgumentException("Unable to read " + blockType + " file from uri `" + uri + "`.", e);
        }
    }

    private static String resolveImageMediaType(byte[] bytes) throws Exception {
        String mediaType = TIKA.detect(bytes);

        if (mediaType == null || mediaType.isBlank() || OCTET_STREAM_MIME_TYPE.equalsIgnoreCase(mediaType)) {
            throw new IllegalArgumentException("Unable to detect IMAGE media type.");
        }
        if (!mediaType.startsWith("image/")) {
            throw new IllegalArgumentException("Invalid IMAGE media type `" + mediaType + "`. It must start with `image/`.");
        }
        return mediaType;
    }
}
