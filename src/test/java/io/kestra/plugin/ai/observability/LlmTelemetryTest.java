package io.kestra.plugin.ai.observability;

import java.nio.charset.StandardCharsets;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.parallel.ResourceLock;

import com.github.tomakehurst.wiremock.WireMockServer;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.MockOpenAI;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;
import io.kestra.plugin.ai.provider.OpenAI;

import jakarta.inject.Inject;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.post;
import static com.github.tomakehurst.wiremock.client.WireMock.postRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo;
import static com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig;
import static org.assertj.core.api.Assertions.assertThat;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class LlmTelemetryTest {
    private static final String TRACE_PARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    private static final String PROMPT = "unique-prompt-token-zz9";

    @RegisterExtension
    static final MockOpenAI llm = new MockOpenAI();

    @Inject
    private TestRunContextFactory runContextFactory;

    private String tracesEndpoint;

    @AfterEach
    void clearOtelEnvironment() {
        LlmTelemetry.clearEnvironment();
        if (tracesEndpoint != null) {
            LlmTelemetry.close(tracesEndpoint);
            tracesEndpoint = null;
        }
    }

    @Test
    void chatCompletionExportsTheLlmSpanWhenKotlpPublishesAnOtlpEndpoint() throws Exception {
        var collector = collector();
        try {
            LlmTelemetry.useEnvironment(Map.of(
                "OTEL_EXPORTER_OTLP_ENDPOINT", collector.baseUrl(),
                "OTEL_SERVICE_NAME", "plugin-ai-test",
                OtlpExportConfig.CAPTURE_MESSAGE_CONTENT, "true"
            ));

            var runContext = runContext();
            var output = chat().run(runContext);

            assertThat(output.getTextOutput()).isEqualTo("Kestra is an open-source orchestration platform.");
            var payload = exportedTrace(collector);
            var body = new String(payload, StandardCharsets.UTF_8);
            assertThat(body).contains("gpt-4o-mini");
            assertThat(body).contains("openai");
            assertThat(body).contains("gen_ai.usage.input_tokens");
            assertThat(body).contains("gen_ai.response.finish_reasons");
            assertThat(body).contains(PROMPT);
            assertThat(body).contains("Kestra is an open-source orchestration platform.");
            // Trace and parent ids are raw bytes in OTLP protobuf, not hex text.
            assertThat(indexOf(payload, HexFormat.of().parseHex("4bf92f3577b34da6a3ce929d0e0e4736"))).isNotNegative();
            assertThat(indexOf(payload, HexFormat.of().parseHex("00f067aa0ba902b7"))).isNotNegative();
            collector.verify(postRequestedFor(urlEqualTo("/v1/traces"))
                .withHeader("Content-Type", com.github.tomakehurst.wiremock.client.WireMock.containing("application/x-protobuf"))
                .withoutHeader("Content-Encoding"));
        } finally {
            collector.stop();
        }
    }

    @Test
    void chatCompletionOmitsPromptTextUnlessCaptureIsEnabled() throws Exception {
        var collector = collector();
        try {
            LlmTelemetry.useEnvironment(Map.of(
                "OTEL_EXPORTER_OTLP_ENDPOINT", collector.baseUrl(),
                OtlpExportConfig.CAPTURE_MESSAGE_CONTENT, "false"
            ));

            assertThat(chat().run(runContext()).getTextOutput()).isNotBlank();

            var body = new String(exportedTrace(collector), StandardCharsets.UTF_8);
            assertThat(body).contains("gpt-4o-mini");
            assertThat(body).contains("gen_ai.request.model");
            assertThat(body).doesNotContain(PROMPT);
            assertThat(body).doesNotContain("Kestra is an open-source orchestration platform.");
        } finally {
            collector.stop();
        }
    }

    @Test
    void chatCompletionDoesNotExportWhenMonitoringIsOff() throws Exception {
        var collector = collector();
        try {
            LlmTelemetry.useEnvironment(Map.of(
                "OTEL_EXPORTER_OTLP_ENDPOINT", collector.baseUrl(),
                "OTEL_TRACES_EXPORTER", "none"
            ));

            assertThat(chat().run(runContext()).getTextOutput()).isNotBlank();

            collector.verify(0, postRequestedFor(urlEqualTo("/v1/traces")));
        } finally {
            collector.stop();
        }
    }

    private WireMockServer collector() {
        var collector = new WireMockServer(wireMockConfig().dynamicPort());
        collector.start();
        collector.stubFor(post(urlEqualTo("/v1/traces")).willReturn(aResponse().withStatus(200)));
        tracesEndpoint = collector.baseUrl() + "/v1/traces";
        return collector;
    }

    private RunContext runContext() {
        var runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", llm.baseUrl()
        ));
        runContext.setTraceParent(TRACE_PARENT);
        return runContext;
    }

    private ChatCompletion chat() {
        return ChatCompletion.builder()
            .id("chat")
            .type(ChatCompletion.class.getName())
            .provider(
                OpenAI.builder()
                    .type(OpenAI.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build()
            )
            .configuration(ChatConfiguration.empty())
            .messages(Property.ofValue(List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content(PROMPT).build()
            )))
            .build();
    }

    private byte[] exportedTrace(WireMockServer collector) {
        LlmTelemetry.flush(tracesEndpoint);
        assertThat(collector.getAllServeEvents()).isNotEmpty();
        return collector.getAllServeEvents().getFirst().getRequest().getBody();
    }

    private static int indexOf(byte[] body, byte[] needle) {
        for (int i = 0; i <= body.length - needle.length; i++) {
            var matches = true;
            for (int j = 0; j < needle.length; j++) {
                if (body[i + j] != needle[j]) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                return i;
            }
        }
        return -1;
    }
}
