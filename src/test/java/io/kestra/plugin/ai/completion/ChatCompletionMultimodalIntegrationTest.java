package io.kestra.plugin.ai.completion;

import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.PdfFileContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.domain.ChatMessageType;
import io.kestra.plugin.ai.domain.ChatMessage;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.instanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

@KestraTest
class ChatCompletionMultimodalIntegrationTest {
    private static final byte[] SAMPLE_PNG_BYTES = Base64.getDecoder().decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5Wm8QAAAAASUVORK5CYII="
    );
    private static final byte[] SAMPLE_PDF_BYTES = (
        "%PDF-1.4\n" +
            "1 0 obj\n" +
            "<< /Type /Catalog >>\n" +
            "endobj\n" +
            "%%EOF\n"
    ).getBytes(StandardCharsets.US_ASCII);

    @Inject
    private RunContextFactory runContextFactory;

    @Test
    void shouldKeepLegacyMessageContentAsTextByDefault() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of());

        ChatCompletion task = ChatCompletion.builder().build();
        List<dev.langchain4j.data.message.ChatMessage> converted = task.convertMessages(
            runContext,
            List.of(
                ChatMessage.builder()
                    .type(ChatMessageType.USER)
                    .content("legacy content")
                    .build()
            )
        );

        UserMessage userMessage = (UserMessage) converted.getFirst();
        assertThat(userMessage.contents(), hasSize(1));
        assertThat(userMessage.contents().getFirst(), instanceOf(TextContent.class));
        assertThat(((TextContent) userMessage.contents().getFirst()).text(), equalTo("legacy content"));
    }

    @Test
    void shouldConvertTextImageAndPdfBlocksFromUploadedFiles() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of());
        String imageUri = upload(runContext, SAMPLE_PNG_BYTES, "tiny.png");
        String pdfUri = upload(runContext, SAMPLE_PDF_BYTES, "sample.pdf");
        String imageBase64 = Base64.getEncoder().encodeToString(SAMPLE_PNG_BYTES);
        String pdfBase64 = Base64.getEncoder().encodeToString(SAMPLE_PDF_BYTES);

        ChatCompletion task = ChatCompletion.builder().build();
        List<dev.langchain4j.data.message.ChatMessage> converted = task.convertMessages(
            runContext,
            List.of(
                ChatMessage.builder()
                    .type(ChatMessageType.USER)
                    .contents(List.of(
                        ChatMessage.ContentBlock.builder()
                            .text("analyze these files")
                            .build(),
                        ChatMessage.ContentBlock.builder()
                            .type(ChatMessage.ContentBlock.Type.IMAGE)
                            .uri(imageUri)
                            .build(),
                        ChatMessage.ContentBlock.builder()
                            .type(ChatMessage.ContentBlock.Type.PDF)
                            .uri(pdfUri)
                            .build()
                    ))
                    .build()
            )
        );

        UserMessage userMessage = (UserMessage) converted.getFirst();
        assertThat(userMessage.contents(), hasSize(3));
        assertThat(userMessage.contents().get(0), instanceOf(TextContent.class));
        assertThat(((TextContent) userMessage.contents().get(0)).text(), equalTo("analyze these files"));

        assertThat(userMessage.contents().get(1), instanceOf(ImageContent.class));
        ImageContent imageContent = (ImageContent) userMessage.contents().get(1);
        assertThat(imageContent.image().base64Data(), equalTo(imageBase64));
        assertThat(imageContent.image().mimeType(), equalTo("image/png"));

        assertThat(userMessage.contents().get(2), instanceOf(PdfFileContent.class));
        PdfFileContent pdfContent = (PdfFileContent) userMessage.contents().get(2);
        assertThat(pdfContent.pdfFile().base64Data(), equalTo(pdfBase64));
        assertThat(pdfContent.pdfFile().mimeType(), equalTo("application/pdf"));
    }

    @Test
    void shouldFailWhenImageUriIsNotKestraStorageUri() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of());

        ChatCompletion task = ChatCompletion.builder().build();
        assertThrows(
            IllegalArgumentException.class,
            () -> task.convertMessages(
                runContext,
                List.of(
                    ChatMessage.builder()
                        .type(ChatMessageType.USER)
                        .contents(List.of(
                            ChatMessage.ContentBlock.builder()
                                .type(ChatMessage.ContentBlock.Type.IMAGE)
                                .uri("https://example.com/image.png")
                                .build()
                        ))
                        .build()
                )
            )
        );
    }

    @Test
    void shouldFailWhenSystemMessageContainsNonTextContentBlock() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of());

        ChatCompletion task = ChatCompletion.builder().build();
        assertThrows(
            IllegalArgumentException.class,
            () -> task.convertMessages(
                runContext,
                List.of(
                    ChatMessage.builder()
                        .type(ChatMessageType.SYSTEM)
                        .contents(List.of(
                            ChatMessage.ContentBlock.builder()
                                .type(ChatMessage.ContentBlock.Type.PDF)
                                .uri("kestra://tmp/dummy")
                                .build()
                        ))
                        .build()
                )
            )
        );
    }

    private String upload(RunContext runContext, byte[] content, String fileName) throws Exception {
        try (InputStream inputStream = new ByteArrayInputStream(content)) {
            return runContext.storage().putFile(inputStream, fileName).toString();
        }
    }
}
