package io.kestra.plugin.ai.agent;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agentic.AgenticServices;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.tool.ToolExecutor;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.models.tasks.runners.ScriptService;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.ListUtils;
import io.kestra.plugin.ai.domain.*;
import io.kestra.plugin.ai.rag.ChatCompletion;
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
public abstract class AbstractAiAgentWorkflow extends Task implements RunnableTask<ChatCompletion.Output> {
    @Schema(title = "Language Model Provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @Schema(title = "Chat configuration")
    @NotNull
    @PluginProperty
    @Builder.Default
    private ChatConfiguration chatConfiguration = ChatConfiguration.empty();

    @Schema(title = "The list of agents")
    @NotNull
    @NotEmpty
    @Valid
    private List<Agent> agents;

    private Property<Map<String, Object>> inputs;

    // TODO allow A2A agents as subagents

    @Override
    public ChatCompletion.Output run(RunContext runContext) throws Exception {
        // TODO output files
        List<ToolProvider> allToolProviders = new ArrayList<>();

        try {
            Object[] assistants = agents.stream()
                .map(throwFunction(agent -> {
                    List<ToolProvider> toolProviders = ListUtils.emptyOnNull(agent.tools);
                    allToolProviders.addAll(toolProviders);
                    return AgenticServices.agentBuilder(SubAgentInterface.class)
                        .chatModel(provider.chatModel(runContext, chatConfiguration))
                        .tools(buildTools(runContext, toolProviders))
                        .outputName(runContext.render(agent.outputName).as(String.class).orElseThrow())
                        .name(runContext.render(agent.name).as(String.class).orElseThrow())
                        .description(runContext.render(agent.description).as(String.class).orElse(null))
                        .systemMessageProvider(throwFunction(memoryId -> runContext.render(agent.systemMessage).as(String.class).orElse(null)))
                        .context(throwFunction(agenticScope -> runContext.render(agent.prompt).as(String.class).orElseThrow()))
                        .build();
                }))
                .toArray();

            Map<String, Object> rInputs = runContext.render(inputs).asMap(String.class, Object.class);

            WorkflowAgent workflowAgent = workflowAgent(runContext, assistants);
            Response<AiMessage> completion = workflowAgent.invoke(rInputs);
            runContext.logger().debug("Generated completion: {}", completion.content());

            return ChatCompletion.Output.builder()
                .completion(completion.content().text())
                .tokenUsage(TokenUsage.from(completion.tokenUsage()))
                .finishReason(completion.finishReason())
                .build();
        } finally {
            allToolProviders.forEach(tool -> tool.close(runContext));
        }
    }

    protected abstract WorkflowAgent workflowAgent(RunContext runContext, Object[] assistants) throws IllegalVariableEvaluationException;

    private Map<ToolSpecification, ToolExecutor> buildTools(RunContext runContext, List<ToolProvider> toolProviders) throws IllegalVariableEvaluationException {
        if (toolProviders.isEmpty()) {
            return Collections.emptyMap();
        }

        Map<ToolSpecification, ToolExecutor> tools = new HashMap<>();
        toolProviders.forEach(throwConsumer(provider -> tools.putAll(provider.tool(runContext, Collections.emptyMap()))));
        return tools;
    }


    protected interface WorkflowAgent {
        @dev.langchain4j.agentic.Agent
        Response<AiMessage> invoke(Map<String, Object> inputs);
    }

    protected interface SubAgentInterface {
        @UserMessage("""
            You are an agent, answer to the query from the context"""
        )
        @dev.langchain4j.agentic.Agent
        Response<AiMessage> invoke();
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

        @Schema(title = "Text prompt", description = "The input prompt for the language model")
        @NotNull
        private Property<String> prompt;

        @Schema(title = "Output name", description = "The name of the output, you can use it in the prompt of other agents via {% raw %}{{outputName}}{% endraw %}")
        @NotNull
        private Property<String> outputName;

        @Schema(title = "Tools that the LLM may use to augment its response")
        private List<ToolProvider> tools;

        // TODO allow providing a different model for each agent

        // TODO allow setting a memory
    }
}
