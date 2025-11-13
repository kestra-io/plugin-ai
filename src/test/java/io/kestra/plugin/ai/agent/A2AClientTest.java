package io.kestra.plugin.ai.agent;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.micronaut.context.ApplicationContext;
import io.micronaut.runtime.server.EmbeddedServer;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class A2AClientTest {
    @Inject
    private TestRunContextFactory runContextFactory;

    @Inject
    private ApplicationContext applicationContext;

    @Test
    void prompt() throws Exception {
        EmbeddedServer embeddedServer = applicationContext.getBean(EmbeddedServer.class);
        embeddedServer.start();

        RunContext runContext = runContextFactory.of(Map.of(
            "agentUrl", "http://localhost:" + embeddedServer.getPort()
        ));

        var agent = A2AClient.builder()
            .serverUrl(Property.ofExpression("{{agentUrl}}"))
            .prompt(Property.ofValue("Hello a2a."))
            .build();

        var output = agent.run(runContext);
        assertThat(output.getTextOutput()).isEqualTo("Hello World");
    }
}