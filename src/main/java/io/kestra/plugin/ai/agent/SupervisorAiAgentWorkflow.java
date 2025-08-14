package io.kestra.plugin.ai.agent;

import dev.langchain4j.agentic.AgenticServices;
import dev.langchain4j.agentic.supervisor.SupervisorResponseStrategy;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

import java.util.List;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(title = "Create a Retrieval Augmented Generation (RAG) pipeline.")
@Plugin(
    examples = {
        @Example(
            full = true,
            title = """
                TODO""",
            code = """
                TODO"""
        ),
    },
    beta = true
)
public class SupervisorAiAgentWorkflow extends AbstractAiAgentWorkflow {
    private Property<Integer> maxAgentsInvocations;

    private Property<SupervisorResponseStrategy> responseStrategy;

    @Override
    protected WorkflowAgent workflowAgent(RunContext runContext, Object[] assistants) throws IllegalVariableEvaluationException {
        return AgenticServices.supervisorBuilder(WorkflowAgent.class)
            .subAgents(assistants)
            .maxAgentsInvocations(runContext.render(maxAgentsInvocations).as(Integer.class).orElse(Integer.MAX_VALUE))
            .responseStrategy(runContext.render(responseStrategy).as(SupervisorResponseStrategy.class).orElse(SupervisorResponseStrategy.LAST))
            .build();
    }
}