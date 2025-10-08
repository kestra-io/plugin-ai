package io.kestra.plugin.ai.tool;

import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.output.FinishReason;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.Label;
import io.kestra.core.models.flows.GenericFlow;
import io.kestra.core.models.property.Property;
import io.kestra.core.repositories.ExecutionRepositoryInterface;
import io.kestra.core.repositories.FlowRepositoryInterface;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;
import io.kestra.plugin.ai.provider.OpenAI;
import io.micronaut.data.model.Pageable;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest(startRunner = true)
class KestraFlowTest {
    @Inject
    private RunContextFactory runContextFactory;

    @Inject
    private FlowRepositoryInterface flowRepository;

    @Inject
    private ExecutionRepositoryInterface executionRepository;

    @Test
    void helloWorld() throws Exception {
        String flowYaml = """
            id: hello-world
            namespace: company.team

            tasks:
              - id: hello
                type: io.kestra.plugin.core.log.Log
                message: Hello World! 🚀
            """;
        var flow = flowRepository.create(GenericFlow.fromYaml(null, flowYaml));

        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1"
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
                KestraFlow.builder().namespace(Property.ofValue("company.team")).flowId(Property.ofValue("hello-world")).description(Property.ofValue("A flow that say Hello World")).build())
            )
            .messages(Property.ofValue(
                List.of(
                    ChatMessage.builder().type(ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                    ChatMessage.builder().type(ChatMessageType.USER).content("I want to execute a flow to say Hello World, please answer with its execution id.").build()
                )))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = chat.run(runContext);
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_flow_company_team_hello-world");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("kestra_flow_company_team_hello-world");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();

        // check that an execution has been created
        var executions = executionRepository.findByFlowId(null, "company.team", "hello-world", Pageable.UNPAGED);
        assertThat(executions).hasSize(1);
        assertThat(output.getTextOutput()).contains(executions.getFirst().getId());

        flowRepository.delete(flow);
        executionRepository.delete(executions.getFirst());
    }

    @Test
    void descriptionFromTheFlow() throws Exception {
        String flowYaml = """
            id: hello-world-with-description
            namespace: company.team
            description: A flow that say Hello World

            tasks:
              - id: hello
                type: io.kestra.plugin.core.log.Log
                message: Hello World! 🚀
            """;
        var flow = flowRepository.create(GenericFlow.fromYaml(null, flowYaml));

        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1"
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
                KestraFlow.builder().namespace(Property.ofValue("company.team")).flowId(Property.ofValue("hello-world-with-description")).build())
            )
            .messages(Property.ofValue(
                List.of(
                    ChatMessage.builder().type(ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                    ChatMessage.builder().type(ChatMessageType.USER).content("I want to execute a flow to say Hello World, please return its response as a valid JSON.").build()
                )))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder()
                .temperature(Property.ofValue(0.1))
                .seed(Property.ofValue(123456789))
                .responseFormat(ChatConfiguration.ResponseFormat.builder().type(Property.ofValue(ResponseFormatType.JSON)).build())
                .build()
            )
            .build();

        var output = chat.run(runContext);
        assertThat(output.getJsonOutput()).isNotEmpty();
        assertThat(output.getJsonOutput()).containsEntry("namespace", "company.team");
        assertThat(output.getJsonOutput()).containsEntry("flowId", "hello-world-with-description");
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_flow_company_team_hello-world-with-description");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("kestra_flow_company_team_hello-world-with-description");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();

        // check that an execution has been created
        var executions = executionRepository.findByFlowId(null, "company.team", "hello-world-with-description", Pageable.UNPAGED);
        assertThat(executions).hasSize(1);
        assertThat(output.getJsonOutput()).containsEntry("id", executions.getFirst().getId());

        flowRepository.delete(flow);
        executionRepository.delete(executions.getFirst());
    }

    @Test
    void inputsAndLabels() throws Exception {
        String flowYaml = """
            id: hello-world-with-input
            namespace: company.team

            labels:
            - key: existing
              value: label

            inputs:
            - id: name
              type: STRING

            tasks:
              - id: hello
                type: io.kestra.plugin.core.log.Log
                message: Hello {{inputs.name}}
            """;
        var flow = flowRepository.create(GenericFlow.fromYaml(null, flowYaml));

        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1"
        ));

        var chat = ChatCompletion.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .tools(
                List.of(KestraFlow.builder().namespace(Property.ofValue("company.team")).flowId(Property.ofValue("hello-world-with-input")).description(Property.ofValue("A flow that say Hello World")).build())
            )
            .messages(Property.ofValue(
                List.of(
                    ChatMessage.builder().type(ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                    ChatMessage.builder().type(ChatMessageType.USER).content("""
                        I want to execute a flow to say Hello World.
                        Call it with the input id 'name' value 'John' and add a label key 'llm' value 'true'.""").build()
                )))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = chat.run(runContext);
        assertThat(output.getTextOutput()).contains("success");
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_flow_company_team_hello-world-with-input");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("kestra_flow_company_team_hello-world-with-input");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();

        // check that an execution has been created
        var executions = executionRepository.findByFlowId(null, "company.team", "hello-world-with-input", Pageable.UNPAGED);
        assertThat(executions).hasSize(1);
        assertThat(executions.getFirst().getLabels()).hasSize(3);
        assertThat(executions.getFirst().getLabels()).contains(
            new Label("existing", "label"),
            new Label("llm", "true")
        );

        flowRepository.delete(flow);
    }

    @Test
    void helloWorldFromLLM() throws Exception {
        String flowYaml = """
            id: hello-world
            namespace: company.team

            tasks:
              - id: hello
                type: io.kestra.plugin.core.log.Log
                message: Hello World! 🚀
            """;
        var flow = flowRepository.create(GenericFlow.fromYaml(null, flowYaml));

        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1"
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
                KestraFlow.builder().build())
            )
            .messages(Property.ofValue(
                List.of(
                    ChatMessage.builder().type(ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                    ChatMessage.builder().type(ChatMessageType.USER).content("I want to execute the flow 'hello-world' from the namespace 'company.team', please answer with its execution id.").build()
                )))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = chat.run(runContext);
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_flow");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("kestra_flow");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();

        // check that an execution has been created
        var executions = executionRepository.findByFlowId(null, "company.team", "hello-world", Pageable.UNPAGED);
        assertThat(executions).hasSize(1);
        assertThat(output.getTextOutput()).contains(executions.getFirst().getId());

        flowRepository.delete(flow);
        executionRepository.delete(executions.getFirst());
    }
}