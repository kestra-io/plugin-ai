package io.kestra.plugin.ai.completion;

import com.github.tomakehurst.wiremock.WireMockServer;
import com.github.tomakehurst.wiremock.client.WireMock;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.provider.OpenAI;
import jakarta.inject.Inject;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

@KestraTest
public class ImageGenerationTest {

    @Inject
    private RunContextFactory runContextFactory;

    private WireMockServer wireMockServer;

    @BeforeEach
    void setUp() {
        // Start WireMock server
        wireMockServer = new WireMockServer();
        wireMockServer.start();
        WireMock.configureFor("localhost", wireMockServer.port());
    }

    @AfterEach
    void tearDown() {
        wireMockServer.stop();
    }

    @Test
    void run() throws Exception {
        // Stub the OpenAI API response
        stubFor(post(urlEqualTo("/v1/images/generations"))
                .willReturn(aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "application/json")
                        .withBody("{\"data\": [{\"url\": \"https://mock-image-url.com/image.png\"}]}")));

        // GIVEN
        RunContext runContext = runContextFactory.of(Map.of(
                "prompt", "Donald Duck in New York, cartoon style",
                "apiKey", "demo",
                "modelName", "dall-e-3",
                "endpoint", "http://localhost:" + wireMockServer.port() + "/v1"
        ));

        ImageGeneration task = ImageGeneration.builder()
                .prompt(Property.ofExpression("{{ prompt }}"))
                .provider(OpenAI.builder()
                        .type(OpenAI.class.getName())
                        .apiKey(Property.ofExpression("{{ apiKey }}"))
                        .modelName(Property.ofExpression("{{ modelName }}"))
                        .baseUrl(Property.ofExpression("{{ endpoint }}"))
                        .build()
                )
                .build();
        // WHEN
        ImageGeneration.Output runOutput = task.run(runContext);

        // THEN
        assertThat(runOutput.getImageUrl(), is("https://mock-image-url.com/image.png"));

        // Verify WireMock interaction
        verify(postRequestedFor(urlEqualTo("/v1/images/generations"))
                .withRequestBody(matchingJsonPath("$.prompt", equalTo("Donald Duck in New York, cartoon style")))
                .withHeader("Authorization", matching("Bearer demo")));
    }

}
