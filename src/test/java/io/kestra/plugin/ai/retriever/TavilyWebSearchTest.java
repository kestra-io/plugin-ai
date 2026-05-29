package io.kestra.plugin.ai.retriever;

import com.github.tomakehurst.wiremock.WireMockServer;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContextFactory;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;

@KestraTest
class TavilyWebSearchTest {
    @Inject
    private RunContextFactory runContextFactory;

    @Test
    void retrieverReturnResultsFromWireMock() throws Exception {
        var wireMock = new WireMockServer(wireMockConfig().dynamicPort());
        try {
            wireMock.start();

            wireMock.stubFor(post(urlEqualTo("/search"))
                .willReturn(aResponse()
                    .withStatus(200)
                    .withHeader("Content-Type", "application/json")
                    .withBody("""
                        {
                          "query": "What is Kestra?",
                          "response_time": 0.5,
                          "results": [
                            {
                              "title": "Kestra - Open Source Orchestration Platform",
                              "url": "https://kestra.io",
                              "content": "Kestra is an open-source workflow orchestration platform.",
                              "score": 0.99
                            }
                          ]
                        }
                        """)));

            var runContext = runContextFactory.of(Map.of());

            var retriever = TavilyWebSearch.builder()
                .apiKey(Property.ofValue("test-api-key"))
                .maxResults(Property.ofValue(3))
                .baseUrl(Property.ofValue(wireMock.baseUrl() + "/"))
                .build();

            ContentRetriever contentRetriever = retriever.contentRetriever(runContext);
            var results = contentRetriever.retrieve(Query.from("What is Kestra?"));

            assertThat(results, is(not(empty())));
            assertThat(results.getFirst().textSegment().text(), containsString("Kestra"));
        } finally {
            wireMock.stop();
        }
    }
}
