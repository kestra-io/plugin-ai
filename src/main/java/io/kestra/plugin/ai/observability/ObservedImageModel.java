package io.kestra.plugin.ai.observability;

import java.util.List;

import dev.langchain4j.data.image.Image;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.output.Response;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;

final class ObservedImageModel implements ImageModel {
    private final ImageModel delegate;
    private final TelemetrySession session;

    ObservedImageModel(ImageModel delegate, TelemetrySession session) {
        this.delegate = delegate;
        this.session = session;
    }

    @Override
    public Response<Image> generate(String prompt) {
        return invoke(prompt, () -> delegate.generate(prompt));
    }

    @Override
    public Response<List<Image>> generate(String prompt, int n) {
        return invoke(prompt, () -> delegate.generate(prompt, n));
    }

    @Override
    public Response<Image> edit(Image image, String prompt) {
        return invoke(prompt, () -> delegate.edit(image, prompt));
    }

    @Override
    public Response<Image> edit(Image image, Image mask, String prompt) {
        return invoke(prompt, () -> delegate.edit(image, mask, prompt));
    }

    private <T> Response<T> invoke(String prompt, ImageCall<T> call) {
        Span span = null;
        try {
            span = GenAiSpans.start(session, "image_generation", session.fallbackModel);
            span.setAttribute(GenAiSpans.OUTPUT_TYPE, "image");
            if (session.captureMessageContent) {
                GenAiSpans.put(span, GenAiSpans.INPUT_MESSAGES, GenAiMessages.textMessages("user", prompt));
            }
        } catch (Exception e) {
            session.logger.debug("Failed to start an image span", e);
            GenAiSpans.end(span);
            return call.call();
        }

        try (Scope ignored = span.makeCurrent()) {
            var response = call.call();
            complete(span, response);
            return response;
        } catch (Exception e) {
            GenAiSpans.fail(session, span, e);
            span = null;
            if (e instanceof RuntimeException runtime) {
                throw runtime;
            }
            throw new IllegalStateException(e);
        } finally {
            GenAiSpans.end(span);
        }
    }

    private void complete(Span span, Response<?> response) {
        try {
            if (response == null) {
                return;
            }
            if (response.tokenUsage() != null) {
                GenAiSpans.put(span, GenAiSpans.INPUT_TOKENS, response.tokenUsage().inputTokenCount());
                GenAiSpans.put(span, GenAiSpans.OUTPUT_TOKENS, response.tokenUsage().outputTokenCount());
            }
            var finishReason = GenAiMessages.finishReason(response.finishReason());
            if (finishReason != null) {
                span.setAttribute(GenAiSpans.FINISH_REASONS, List.of(finishReason));
            }
            if (!session.captureMessageContent || response.content() == null) {
                return;
            }
            if (response.content() instanceof Image image) {
                GenAiSpans.put(span, GenAiSpans.OUTPUT_MESSAGES, GenAiMessages.textMessages("assistant", imageUrl(image)));
            } else if (response.content() instanceof List<?> images) {
                var urls = images.stream()
                    .filter(Image.class::isInstance)
                    .map(Image.class::cast)
                    .map(ObservedImageModel::imageUrl)
                    .filter(url -> url != null && !url.isBlank())
                    .toList();
                if (!urls.isEmpty()) {
                    GenAiSpans.put(span, GenAiSpans.OUTPUT_MESSAGES, GenAiMessages.textMessages("assistant", String.join("\n", urls)));
                }
            }
        } catch (Exception e) {
            session.logger.debug("Failed to complete an image span", e);
        }
    }

    private static String imageUrl(Image image) {
        if (image == null || image.url() == null) {
            return null;
        }
        var url = image.url().toString();
        return url.startsWith("data:") ? null : url;
    }

    @FunctionalInterface
    private interface ImageCall<T> {
        Response<T> call();
    }
}
