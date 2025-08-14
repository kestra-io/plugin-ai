package io.kestra.plugin.ai.agent;

import dev.langchain4j.agentic.AgenticServices;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.TruthUtils;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

import java.util.List;

import static io.kestra.core.utils.Rethrow.throwPredicate;

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
public class LoopingAiAgentWorkflow extends AbstractAiAgentWorkflow {
    private Property<Integer> maxIterations;

    @NotNull
    private Property<String> exitCondition;

    @Override
    protected WorkflowAgent workflowAgent(RunContext runContext, Object[] assistants) throws IllegalVariableEvaluationException {
        return AgenticServices.loopBuilder(WorkflowAgent.class)
            .subAgents(assistants)
            .maxIterations(runContext.render(maxIterations).as(Integer.class).orElse(Integer.MAX_VALUE))
            .exitCondition(throwPredicate(agenticScope -> TruthUtils.isTruthy(runContext.render(exitCondition).as(String.class, agenticScope.state()).orElseThrow())))
            .build();
    }
}
