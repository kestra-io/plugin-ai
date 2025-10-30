package io.kestra.plugin.ai.tool;

import dev.langchain4j.model.output.FinishReason;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;
import io.kestra.plugin.ai.provider.OpenAI;
import io.micronaut.context.ApplicationContext;
import io.micronaut.runtime.server.EmbeddedServer;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest(startRunner = true)
class A2AClientTest {
    @Inject
    private RunContextFactory runContextFactory;

    @Inject
    private ApplicationContext applicationContext;

    @Test
    void agentTest() throws Exception {
        EmbeddedServer embeddedServer = applicationContext.getBean(EmbeddedServer.class);
        embeddedServer.start();

        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1",
            "agentUrl", "http://localhost:" + embeddedServer.getPort()
        ));

        var chat = ChatCompletion.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .tools(List.of(
                A2AClient.builder()
                    .serverUrl(Property.ofExpression("{{agentUrl}}"))
                    .description(Property.ofValue("An AI agent expert on translation"))
                    .build()
            ))
            .messages(Property.ofValue(
                List.of(
                    ChatMessage.builder().type(ChatMessageType.SYSTEM).content("You are an AI agent, summarize the user message then translate it in french using the provided tool.").build(),
                    ChatMessage.builder().type(ChatMessageType.USER).content("""
                        Each flow can produce outputs that can be consumed by other flows. This is a list property, so that your flow can produce as many outputs as you need.
                        Each output needs to have an `id` (the name of the output), a `type` (the same types you know from inputs e.g., STRING, URI or JSON), and `value`, which is the actual output value that will be stored in internal storage and passed to other flows when needed.""").build()
                )))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = chat.run(runContext);
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_a2a_agent_tool");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("kestra_a2a_agent_tool");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();
    }
}