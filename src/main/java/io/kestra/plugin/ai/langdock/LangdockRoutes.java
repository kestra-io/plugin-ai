package io.kestra.plugin.ai.langdock;

import java.util.Locale;

/**
 * Builds the Langdock Completion API base URLs shared by the {@code provider.Langdock} chat/embedding
 * models and the {@code langdock.ListModels} task, so both build a URL for a given region/family the
 * same way. See <a href="https://docs.langdock.com/en/developer/completion-api/completion-overview">the
 * Completion API overview</a>.
 */
public final class LangdockRoutes {
    private static final String DEFAULT_HOST = "https://api.langdock.com";

    private LangdockRoutes() {
    }

    /**
     * The chat route root for a given region/family, e.g. {@code https://api.langdock.com/openai/eu/v1}
     * or {@code https://api.langdock.com/anthropic/eu/v1/}. The Anthropic route needs a trailing slash
     * because {@code DefaultAnthropicClient} appends {@code messages} directly to the base URL.
     */
    public static String chatBaseUrl(LangdockRegion region, LangdockModelFamily family) {
        String regionSegment = region.name().toLowerCase(Locale.ROOT);
        return switch (family) {
            case OPENAI -> DEFAULT_HOST + "/openai/" + regionSegment + "/v1";
            case ANTHROPIC -> DEFAULT_HOST + "/anthropic/" + regionSegment + "/v1/";
        };
    }

    /**
     * The embeddings route root for a given region. Embeddings are only exposed on the OpenAI route,
     * whatever {@link LangdockModelFamily} the chat model is configured with.
     */
    public static String embeddingsBaseUrl(LangdockRegion region) {
        return DEFAULT_HOST + "/openai/" + region.name().toLowerCase(Locale.ROOT) + "/v1";
    }

    /**
     * Normalizes the trailing slash of a user-supplied {@code baseUrl} (dedicated deployment) to match
     * what each family's HTTP client expects. The path itself is never rewritten.
     */
    public static String normalizeBaseUrl(String baseUrl, LangdockModelFamily family) {
        String trimmed = baseUrl.strip();
        return switch (family) {
            case ANTHROPIC -> trimmed.endsWith("/") ? trimmed : trimmed + "/";
            case OPENAI -> trimmed.endsWith("/") ? trimmed.substring(0, trimmed.length() - 1) : trimmed;
        };
    }
}
