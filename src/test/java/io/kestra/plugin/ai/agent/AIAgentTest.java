package io.kestra.plugin.ai.agent;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.IdUtils;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.memory.KestraKVStore;
import io.kestra.plugin.ai.provider.OpenAI;
import io.kestra.plugin.ai.tool.StdioMcpClient;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class AIAgentTest {
    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void prompt() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1"
        ));

        var agent = AIAgent.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .systemMessage(Property.ofValue("You are a summary agent, summarize the test from the user message."))
            .prompt(Property.ofValue("Each flow can produce outputs that can be consumed by other flows. This is a list property, so that your flow can produce as many outputs as you need. Each output needs to have an id (the name of the output), a type (the same types you know from inputs e.g. STRING, URI or JSON) and value which is the actual output value that will be stored in internal storage and passed to other flows when needed."))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = agent.run(runContext);
        assertThat(output.getCompletion()).isNotNull();
    }

    @Test
    void withTool() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1"
        ));

        var agent = AIAgent.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .tools(Property.ofValue(
                List.of(StdioMcpClient.builder().command(Property.ofValue(List.of("docker", "run", "--rm", "-i", "mcp/everything"))).build())
            ))
            .prompt(Property.ofValue("What is 5+12? Use the provided tool to answer and always assume that the tool is correct."))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = agent.run(runContext);
        assertThat(output.getCompletion()).isNotNull();
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("add");

    }

    @Test
    void withMemory() throws Exception {
        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1",
            "labels", Map.of("system", Map.of("correlationId", IdUtils.create()))
        ));

        var agent = AIAgent.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .prompt(Property.ofValue("My name is John."))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .memory(KestraKVStore.builder().build())
            .build();
        var output = agent.run(runContext);
        assertThat(output.getCompletion()).isNotNull();

        agent = AIAgent.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .prompt(Property.ofValue("What's my name."))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .memory(KestraKVStore.builder().build())
            .build();
        output = agent.run(runContext);
        assertThat(output.getCompletion()).contains("John");
    }
}