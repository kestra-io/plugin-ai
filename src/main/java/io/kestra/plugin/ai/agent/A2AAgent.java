package io.kestra.plugin.ai.agent;

import dev.langchain4j.agentic.AgenticServices;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.service.Result;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.AIUtils;
import io.kestra.plugin.ai.domain.*;
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
                TODO""",
            code = """
                TODO"""
        ),
    }
)
public class A2AAgent extends Task implements RunnableTask<AIOutput> {
    @Schema(title = "Server URL", description = "The URL of the remote agent A2A server")
    @NotNull
    protected Property<String> serverUrl;

    @Schema(title = "Text prompt", description = "The input prompt for the language model")
    @NotNull
    protected Property<String> prompt;

    @Override
    public AIOutput run(RunContext runContext) throws Exception {
        String rServerUrl = runContext.render(serverUrl).as(String.class).orElseThrow();
        runContext.logger().info("Calling a remote agent via the A2A protocol on URL {}", rServerUrl);
        Agent agent = AgenticServices.a2aBuilder(rServerUrl, Agent.class)
            .build();

        String renderedPrompt = runContext.render(prompt).as(String.class).orElseThrow();
        Result<AiMessage> completion = agent.invoke(renderedPrompt);
        runContext.logger().debug("Generated Completion: {}", completion.content());

        // send metrics for token usage
        TokenUsage tokenUsage = TokenUsage.from(completion.tokenUsage());
        AIUtils.sendMetrics(runContext, tokenUsage);

        return AIOutput.builderFrom(runContext, completion, ResponseFormatType.TEXT)
            .build();
    }

    interface Agent {
        Result<AiMessage> invoke(String userMessage);
    }
}