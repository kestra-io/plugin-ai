package io.kestra.plugin.ai.langdock;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class LangdockRoutesTest {

    @Test
    void chatBaseUrl_shouldBuildOpenAiRoute() {
        assertThat(LangdockRoutes.chatBaseUrl(LangdockRegion.EU, LangdockModelFamily.OPENAI))
            .isEqualTo("https://api.langdock.com/openai/eu/v1");
        assertThat(LangdockRoutes.chatBaseUrl(LangdockRegion.US, LangdockModelFamily.OPENAI))
            .isEqualTo("https://api.langdock.com/openai/us/v1");
    }

    @Test
    void chatBaseUrl_shouldBuildAnthropicRouteWithTrailingSlash() {
        assertThat(LangdockRoutes.chatBaseUrl(LangdockRegion.EU, LangdockModelFamily.ANTHROPIC))
            .isEqualTo("https://api.langdock.com/anthropic/eu/v1/");
        assertThat(LangdockRoutes.chatBaseUrl(LangdockRegion.US, LangdockModelFamily.ANTHROPIC))
            .isEqualTo("https://api.langdock.com/anthropic/us/v1/");
    }

    @Test
    void embeddingsBaseUrl_shouldAlwaysUseOpenAiRoute() {
        assertThat(LangdockRoutes.embeddingsBaseUrl(LangdockRegion.EU))
            .isEqualTo("https://api.langdock.com/openai/eu/v1");
        assertThat(LangdockRoutes.embeddingsBaseUrl(LangdockRegion.US))
            .isEqualTo("https://api.langdock.com/openai/us/v1");
    }

    @Test
    void normalizeBaseUrl_shouldEnsureAnthropicTrailingSlash() {
        assertThat(LangdockRoutes.normalizeBaseUrl("https://acme.langdock.com/api/public/anthropic/eu/v1", LangdockModelFamily.ANTHROPIC))
            .isEqualTo("https://acme.langdock.com/api/public/anthropic/eu/v1/");
        assertThat(LangdockRoutes.normalizeBaseUrl("https://acme.langdock.com/api/public/anthropic/eu/v1/", LangdockModelFamily.ANTHROPIC))
            .isEqualTo("https://acme.langdock.com/api/public/anthropic/eu/v1/");
    }

    @Test
    void normalizeBaseUrl_shouldStripOpenAiTrailingSlash() {
        assertThat(LangdockRoutes.normalizeBaseUrl("https://acme.langdock.com/api/public/openai/eu/v1/", LangdockModelFamily.OPENAI))
            .isEqualTo("https://acme.langdock.com/api/public/openai/eu/v1");
        assertThat(LangdockRoutes.normalizeBaseUrl("https://acme.langdock.com/api/public/openai/eu/v1", LangdockModelFamily.OPENAI))
            .isEqualTo("https://acme.langdock.com/api/public/openai/eu/v1");
    }
}
