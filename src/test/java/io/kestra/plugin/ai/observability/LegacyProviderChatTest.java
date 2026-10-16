package io.kestra.plugin.ai.observability;

import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.provider.LocalAI;
import io.kestra.plugin.ai.provider.WorkersAI;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class LegacyProviderChatTest {
    private static final String COLLECTOR = "http://127.0.0.1:9";

    @Inject
    private TestRunContextFactory runContextFactory;

    @AfterEach
    void clearOtelEnvironment() {
        LlmTelemetry.clearEnvironment();
        LlmTelemetry.close(COLLECTOR + "/v1/traces");
    }

    @Test
    void localAiChatReachesTheServerWhenNoOtlpEndpointIsConfigured() throws Exception {
        LlmTelemetry.useEnvironment(Map.of());

        var model = localAi().chatModel(runContextFactory.of(), ChatConfiguration.empty());

        assertThat(model).isNotInstanceOf(ObservedChatModel.class);
        assertThatThrownBy(() -> model.chat(request()))
            .isInstanceOf(RuntimeException.class)
            .hasMessageContaining("java.net.ConnectException");
    }

    @Test
    void localAiChatReachesTheServerWhenAnOtlpEndpointIsConfigured() throws Exception {
        LlmTelemetry.useEnvironment(Map.of("OTEL_EXPORTER_OTLP_ENDPOINT", COLLECTOR));

        var model = localAi().chatModel(runContextFactory.of(), ChatConfiguration.empty());

        assertThat(model).isInstanceOf(ObservedChatModel.class);
        assertThatThrownBy(() -> model.chat(request()))
            .isInstanceOf(RuntimeException.class)
            .hasMessageContaining("java.net.ConnectException");
    }

    @Test
    void workersAiChatReachesTheServerWhenAnOtlpEndpointIsConfigured() throws Exception {
        LlmTelemetry.useEnvironment(Map.of("OTEL_EXPORTER_OTLP_ENDPOINT", COLLECTOR));

        var model = workersAi().chatModel(runContextFactory.of(), ChatConfiguration.empty());

        assertThat(model).isInstanceOf(ObservedChatModel.class);
        assertThatThrownBy(() -> model.chat(request()))
            .isInstanceOf(RuntimeException.class)
            .hasMessageContaining("WORKERS_AI_ACCOUNT_ID");
    }

    private static LocalAI localAi() {
        return LocalAI.builder()
            .type(LocalAI.class.getName())
            .modelName(Property.ofValue("gemma-3-1b-it"))
            .baseUrl(Property.ofValue("http://127.0.0.1:9/v1"))
            .build();
    }

    private static WorkersAI workersAi() {
        return WorkersAI.builder()
            .type(WorkersAI.class.getName())
            .modelName(Property.ofValue("@cf/meta/llama-2-7b-chat-int8"))
            .apiKey(Property.ofValue("WORKERS_AI_API_KEY"))
            .accountId(Property.ofValue("WORKERS_AI_ACCOUNT_ID"))
            .build();
    }

    private static ChatRequest request() {
        return ChatRequest.builder().messages(UserMessage.from("Hello, my name is John")).build();
    }
}
