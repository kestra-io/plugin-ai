package io.kestra.plugin.ai.mcp;

import java.util.Map;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
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

@Execution(ExecutionMode.SAME_THREAD)
@ResourceLock("kestra-h2-flyway")
@KestraTest
class ListToolsTest {
    @Inject
    private RunContextFactory runContextFactory;

    private static GenericContainer<?> mcpContainer;

    @BeforeAll
    static void setUp() {
        mcpContainer = new GenericContainer<>(DockerImageName.parse("mcp/everything"))
            .withExposedPorts(3001)
            .withCommand("node", "dist/sse.js")
            // the exposed port accepts connections slightly before the SSE endpoint is actually wired up
            .waitingFor(Wait.forLogMessage(".*Server is running on port.*\\n", 1));
        mcpContainer.start();
    }

    @AfterAll
    static void tearDown() {
        if (mcpContainer != null) {
            mcpContainer.stop();
        }
    }

    @Test
    void shouldListToolsFromServer() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of());

        ListTools task = ListTools.builder()
            .url(Property.ofValue("http://localhost:" + mcpContainer.getMappedPort(3001) + "/sse"))
            .transport(Property.ofValue(AbstractMcpTask.Transport.SSE))
            .build();

        ListTools.Output output = task.run(runContext);

        assertThat(output.getCount()).isEqualTo(output.getTools().size());
        assertThat(output.getTools()).extracting(ListTools.ToolDefinition::name).contains("add", "echo");
        assertThat(output.getTools()).allSatisfy(tool -> assertThat(tool.parameters()).isNotNull());
    }
}
