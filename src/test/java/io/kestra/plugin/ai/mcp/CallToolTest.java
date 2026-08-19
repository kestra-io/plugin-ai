package io.kestra.plugin.ai.mcp;

import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.utility.DockerImageName;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@Execution(ExecutionMode.SAME_THREAD)
@ResourceLock("kestra-h2-flyway")
@KestraTest
class CallToolTest {
    @Inject
    private RunContextFactory runContextFactory;

    private GenericContainer<?> mcpContainer;

    // A fresh container per test: the "everything" SSE server does not reliably serve a second sequential session on the same instance.
    @BeforeEach
    void setUp() {
        mcpContainer = new GenericContainer<>(DockerImageName.parse("mcp/everything"))
            .withExposedPorts(3001)
            .withCommand("node", "dist/sse.js")
            .waitingFor(Wait.forLogMessage(".*Server is running on port.*\\n", 1));
        mcpContainer.start();
    }

    @AfterEach
    void tearDown() {
        if (mcpContainer != null) {
            mcpContainer.stop();
        }
    }

    private String mcpSseUrl() {
        return "http://localhost:" + mcpContainer.getMappedPort(3001) + "/sse";
    }

    @Test
    void shouldReturnResultWhenCallingTool() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of());

        CallTool task = CallTool.builder()
            .url(Property.ofValue(mcpSseUrl()))
            .transport(Property.ofValue(AbstractMcpTask.Transport.SSE))
            .tool(Property.ofValue("add"))
            .arguments(Property.ofValue(Map.of("a", 5, "b", 12)))
            .build();

        CallTool.Output output = task.run(runContext);

        assertThat(output.getResult()).contains("17");
        assertThat(output.getIsError()).isFalse();
        assertThat(output.getErrorMessage()).isNull();
    }

    @Test
    void shouldRenderPropertiesFromExpressions() throws Exception {
        RunContext runContext = runContextFactory.of(
            Map.of(
                "mcpUrl", mcpSseUrl(),
                "toolName", "add"
            )
        );

        CallTool task = CallTool.builder()
            .url(Property.ofExpression("{{ mcpUrl }}"))
            .transport(Property.ofValue(AbstractMcpTask.Transport.SSE))
            .tool(Property.ofExpression("{{ toolName }}"))
            .arguments(Property.ofValue(Map.of("a", 3, "b", 4)))
            .build();

        CallTool.Output output = task.run(runContext);

        assertThat(output.getResult()).contains("7");
        assertThat(output.getIsError()).isFalse();
    }

    @Test
    void shouldThrowWhenToolFails() {
        RunContext runContext = runContextFactory.of(Map.of());

        CallTool task = CallTool.builder()
            .url(Property.ofValue(mcpSseUrl()))
            .transport(Property.ofValue(AbstractMcpTask.Transport.SSE))
            .tool(Property.ofValue("unknown_tool"))
            .build();

        assertThatThrownBy(() -> task.run(runContext)).isInstanceOf(RuntimeException.class);
    }

    @Test
    void shouldReportErrorWhenFailOnToolErrorIsFalse() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of());

        CallTool task = CallTool.builder()
            .url(Property.ofValue(mcpSseUrl()))
            .transport(Property.ofValue(AbstractMcpTask.Transport.SSE))
            .tool(Property.ofValue("unknown_tool"))
            .failOnToolError(Property.ofValue(false))
            .build();

        CallTool.Output output = task.run(runContext);

        assertThat(output.getIsError()).isTrue();
        assertThat(output.getErrorMessage()).isNotNull();
    }
}
