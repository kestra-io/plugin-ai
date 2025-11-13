package io.kestra.plugin.ai.agent;

import dev.langchain4j.agentic.AgenticServices;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(title = "Call a remote AI agent via the A2A protocol.")
@Plugin(
    examples = {
        @Example(
            full = true,
            title = """
                Call a remote AI agent via the A2A protocol.""",
            code = """
                id: a2a_agent
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.A2AClient
                    serverUrl: "http://localhost:10000"
                    prompt: "Summarize the following content: {{ inputs.prompt }}\""""
        ),
    }
)
public class A2AClient extends Task implements RunnableTask<A2AClient.Output> {
    @Schema(title = "Server URL", description = "The URL of the remote agent A2A server")
    @NotNull
    protected Property<String> serverUrl;

    @Schema(title = "Text prompt", description = "The input prompt for the language model")
    @NotNull
    protected Property<String> prompt;

    @Override
    public Output run(RunContext runContext) throws Exception {
        String rServerUrl = runContext.render(serverUrl).as(String.class).orElseThrow();
        runContext.logger().info("Calling a remote agent via the A2A protocol on URL {}", rServerUrl);
        Agent agent = AgenticServices.a2aBuilder(rServerUrl, Agent.class)
            .build();

        String renderedPrompt = runContext.render(prompt).as(String.class).orElseThrow();
        String completion = agent.invoke(renderedPrompt);
        runContext.logger().debug("Generated Completion: {}", completion);

        return Output.builder()
            .textOutput(completion)
            .build();
    }

    interface Agent {
        String invoke(String userMessage);
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(
            title = "Remote agent output"
        )
        private String textOutput;
    }
}