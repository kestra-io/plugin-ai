package io.kestra.plugin.ai.tool;

import dev.langchain4j.model.output.FinishReason;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.provider.OpenAI;
import jakarta.inject.Inject;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;

import java.time.Duration;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class SseMcpClientTest {
    @Inject
    private RunContextFactory runContextFactory;

    public static GenericContainer<?> mcpContainer;

    @BeforeAll
    public static void setUp() throws Exception {
        // Docker image
        DockerImageName dockerImageName = DockerImageName.parse("mcp/everything");

        // Create the container
        mcpContainer = new GenericContainer<>(dockerImageName)
            .withExposedPorts(3001)
            .withCommand("node", "dist/sse.js");

        // Start the container
        mcpContainer.start();
    }

    @AfterAll
    static void tearDown() {
        if (mcpContainer != null) {
            mcpContainer.stop();
        }
    }

    @Test
    void chat() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1",
            "mcpSseUrl", "http://localhost:" + mcpContainer.getMappedPort(3001) + "/sse"
        ));

        var chat = ChatCompletion.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .tools(List.of(
                SseMcpClient.builder().sseUrl(Property.ofExpression("{{mcpSseUrl}}")).timeout(Property.ofValue(Duration.ofSeconds(60))).build())
            )
            .messages(Property.ofValue(
                List.of(ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.USER).content("What is 5+12? Use the provided tool to answer and always assume that the tool is correct.").build()
                )))
            .build();

        var output = chat.run(runContext);
        assertThat(output.getCompletion()).contains("17");
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("add");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("add");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();
    }
}