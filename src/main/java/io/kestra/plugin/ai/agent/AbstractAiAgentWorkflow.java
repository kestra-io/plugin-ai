package io.kestra.plugin.ai.agent;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agentic.AgenticServices;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.service.Result;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;
import dev.langchain4j.service.tool.ToolExecutor;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.OutputFilesInterface;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.ListUtils;
import io.kestra.plugin.ai.AIUtils;
import io.kestra.plugin.ai.domain.*;
import io.kestra.plugin.ai.provider.TimingChatModelListener;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;

import java.util.*;

import static io.kestra.core.utils.Rethrow.throwConsumer;
import static io.kestra.core.utils.Rethrow.throwFunction;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
public abstract class AbstractAiAgentWorkflow extends Task implements RunnableTask<AIOutput>, OutputFilesInterface {
    @Schema(title = "Language Model Provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @Schema(title = "Chat configuration")
    @NotNull
    @PluginProperty
    @Builder.Default
    private ChatConfiguration configuration = ChatConfiguration.empty();

    @Schema(title = "The list of agents")
    @NotNull
    @NotEmpty
    @Valid
    private List<Agent> agents;

    @Schema(title = "Inputs", description = "Inputs passed to each agents, you can use them in an agent prompt using `{{inputName}}`.")
    private Property<Map<String, Object>> inputs;

    private Property<List<String>> outputFiles;

    // TODO allow A2A agents as subagents

    @Override
    public AIOutput run(RunContext runContext) throws Exception {
        List<ToolProvider> allToolProviders = new ArrayList<>();
        Map<String, Object> rInputs = runContext.render(inputs).asMap(String.class, Object.class);

        try {
            Object[] assistants = agents.stream()
                .map(throwFunction(agent -> {
                    List<ToolProvider> toolProviders = ListUtils.emptyOnNull(agent.tools);
                    allToolProviders.addAll(toolProviders);
                    return AgenticServices.agentBuilder(SubAgentInterface.class)
                        .chatModel(provider.chatModel(runContext, configuration))
                        .tools(buildTools(runContext, toolProviders))
                        .outputKey(runContext.render(agent.outputName).as(String.class).orElseThrow())
                        .name(runContext.render(agent.name).as(String.class).orElseThrow())
                        .description(runContext.render(agent.description).as(String.class).orElse(null))
                        .systemMessageProvider(throwFunction(memoryId -> runContext.render(agent.systemMessage).as(String.class).orElse(null)))
                        .defaultKeyValue("prompt", agent.prompt)
                        .build();
                    }
                ))
                .toArray();

            WorkflowAgent workflowAgent = workflowAgent(runContext, assistants, runContext.render(agents.getLast().outputName).as(String.class).orElseThrow());
            Result<AiMessage> completion = workflowAgent.invoke(rInputs);
            runContext.logger().debug("Generated completion: {}", completion.content());

            // send metrics for token usage
            TokenUsage tokenUsage = TokenUsage.from(completion.tokenUsage());
            AIUtils.sendMetrics(runContext, tokenUsage);

            return AIOutput.builderFrom(runContext, completion, configuration.computeResponseFormat(runContext).type())
                .outputFiles(AIUtils.gatherOutputFiles(outputFiles, runContext))
                .build();
        } finally {
            allToolProviders.forEach(tool -> tool.close(runContext));

            TimingChatModelListener.clear();
        }
    }

    protected abstract WorkflowAgent workflowAgent(RunContext runContext, Object[] assistants, String outputName) throws IllegalVariableEvaluationException;

    private Map<ToolSpecification, ToolExecutor> buildTools(RunContext runContext, List<ToolProvider> toolProviders) throws Exception {
        if (toolProviders.isEmpty()) {
            return Collections.emptyMap();
        }

        Map<ToolSpecification, ToolExecutor> tools = new HashMap<>();
        toolProviders.forEach(throwConsumer(provider -> tools.putAll(provider.tool(runContext, Collections.emptyMap()))));
        return tools;
    }


    protected interface WorkflowAgent {
        @dev.langchain4j.agentic.Agent
        Result<AiMessage> invoke(Map<String, Object> inputs);
    }

    protected interface SubAgentInterface {
        @UserMessage("{{prompt}}")
        @dev.langchain4j.agentic.Agent
        Result<AiMessage> invoke(@V("prompt") String prompt);
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Generated text completion", description = "The result of the text completion")
        private String completion;

        @Schema(title = "Token usage")
        private TokenUsage tokenUsage;

        @Schema(title = "Finish reason")
        private FinishReason finishReason;
    }

    @Getter
    @Builder
    public static class Agent {
        @Schema(title = "Agent name")
        @NotNull
        private Property<String> name;

        @Schema(title = "Agent description")
        private Property<String> description;

        @Schema(title = "System message", description = "The system message for the language model")
        private Property<String> systemMessage;

        @Schema(
            title = "Text prompt, use inputs from the agent workflow in the form of `{{inputName}}` or output of other agents inside the workflow with `{{outputName}}`.",
            description = """
                The input prompt for the language model.
                It is not dynamically rendered by Kestra as for other task properties, but you can use inputs from the task's `inputs` property or previous agents outputs.
                To use task's inputs or previous agents outputs reference them with their name like `{{inputName}}` or `{{outputName}}`."""
        )
        @NotNull
        @PluginProperty
        private String prompt;

        @Schema(title = "Output name", description = "The name of the output, you can use it in the prompt of other agents via {{outputName}}")
        @NotNull
        private Property<String> outputName;

        @Schema(title = "Tools that the LLM may use to augment its response")
        private List<ToolProvider> tools;

        // TODO allow providing a different model for each agent

        // TODO allow setting a memory
    }
}
