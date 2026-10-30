package io.kestra.plugin.ai.tool.internal;

import java.net.URI;
import java.time.Duration;

import dev.langchain4j.community.web.search.firecrawl.FirecrawlWebSearchEngine;

public final class FirecrawlWebSearchEngines {
    private FirecrawlWebSearchEngines() {
    }

    public static FirecrawlWebSearchEngine buildEngine(String rApiKey, Boolean rScrapeContent, String rLocation, String rTimeBasedFilter, String rBaseUrl, Duration rTimeout) {
        if (rApiKey == null) {
            throw new IllegalArgumentException("`apiKey` is required: set it to your Firecrawl API key, stored as a Kestra secret");
        }
        var builder = FirecrawlWebSearchEngine.builder()
            .apiKey(rApiKey)
            .scrapeContent(Boolean.TRUE.equals(rScrapeContent))
            .location(rLocation)
            .timeBasedFilter(rTimeBasedFilter);
        if (rTimeout != null) {
            builder.timeout(rTimeout);
        }
        if (rBaseUrl != null) {
            var scheme = URI.create(rBaseUrl).getScheme();
            if (!"http".equalsIgnoreCase(scheme) && !"https".equalsIgnoreCase(scheme)) {
                throw new IllegalArgumentException("`baseUrl` must be an http or https URL, got " + rBaseUrl);
            }
            builder.baseUrl(rBaseUrl);
        }
        return builder.build();
    }

    public static int checkMaxResults(Integer rMaxResults) {
        if (rMaxResults == null || rMaxResults < 1 || rMaxResults > 100) {
            throw new IllegalArgumentException("`maxResults` must be between 1 and 100, got " + rMaxResults);
        }
        return rMaxResults;
    }
}
