package io.kestra.plugin.ai.agent;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class A2AAgentTest {
    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void prompt() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "agentUrl", "http://localhost:10000"
        ));

        var agent = A2AAgent.builder()
            .serverUrl(Property.ofExpression("{{agentUrl}}"))
            .prompt(Property.ofValue("how much is 10 USD in INR?"))
            .build();

        var output = agent.run(runContext);
        System.out.println(output.getTextOutput());
        assertThat(output.getTextOutput()).isNotNull();
    }
}