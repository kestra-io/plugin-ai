package io.kestra.plugin.ai.agent;

import dev.langchain4j.agentic.AgenticServices;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
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
public class SequentialAiAgentWorkflow extends AbstractAiAgentWorkflow {
    @Override
    protected WorkflowAgent workflowAgent(RunContext runContext, Object[] assistants) {
        return AgenticServices.sequenceBuilder(WorkflowAgent.class)
            .subAgents(assistants)
            .build();
    }
}
