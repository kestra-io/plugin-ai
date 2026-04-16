package io.kestra.plugin.ai.tool;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;
import io.kestra.plugin.ai.provider.OpenAI;

import com.github.tomakehurst.wiremock.WireMockServer;
import com.github.tomakehurst.wiremock.core.WireMockConfiguration;
import dev.langchain4j.model.output.FinishReason;
import jakarta.inject.Inject;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static org.assertj.core.api.Assertions.assertThat;

@KestraTest(startRunner = true)
class KestraFlowTest {
    @Inject
    private RunContextFactory runContextFactory;

    private WireMockServer wireMock;

    @BeforeEach
    void setUp() {
        wireMock = new WireMockServer(WireMockConfiguration.wireMockConfig().dynamicPort());
        wireMock.start();
    }

    @AfterEach
    void tearDown() {
        wireMock.stop();
    }

    private String flowJson(String namespace, String flowId, Integer revision, String description, String inputsJson) {
        return """
            {"id":"%s","namespace":"%s","revision":%d%s%s}
            """.formatted(
            flowId,
            namespace,
            revision != null ? revision : 1,
            description != null ? ",\"description\":\"" + description + "\"" : "",
            inputsJson != null ? ",\"inputs\":" + inputsJson : ""
        );
    }

    private String executionJson(String id, String namespace, String flowId) {
        return """
            {"id":"%s","namespace":"%s","flowId":"%s","state":{"current":"CREATED"}}
            """.formatted(id, namespace, flowId);
    }

    @Test
    void helloWorld() throws Exception {
        wireMock.stubFor(get(urlPathMatching("/api/v1/.*/flows/company\\.team/hello-world"))
            .willReturn(okJson(flowJson("company.team", "hello-world", 1, null, null))));
        wireMock.stubFor(post(urlPathMatching("/api/v1/.*/executions/company\\.team/hello-world"))
            .willReturn(okJson(executionJson("test-exec-123", "company.team", "hello-world"))));

        RunContext runContext = runContextFactory.of(
            Map.of(
                "apiKey", "demo",
                "modelName", "gpt-4o-mini",
                "baseUrl", "http://langchain4j.dev/demo/openai/v1"
            )
        );

        var chat = ChatCompletion.builder()
            .provider(
                OpenAI.builder()
                    .type(OpenAI.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build()
            )
            .tools(
                List.of(
                    KestraFlow.builder()
                        .namespace(Property.ofValue("company.team"))
                        .flowId(Property.ofValue("hello-world"))
                        .description(Property.ofValue("A flow that say Hello World"))
                        .kestraUrl(Property.ofValue("http://localhost:" + wireMock.port()))
                        .build()
                )
            )
            .messages(
                Property.ofValue(
                    List.of(
                        ChatMessage.builder().type(ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                        ChatMessage.builder().type(ChatMessageType.USER).content("I want to execute a flow to say Hello World, please answer with its execution id.").build()
                    )
                )
            )
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

        wireMock.verify(postRequestedFor(urlPathMatching("/api/v1/.*/executions/company\\.team/hello-world")));
        assertThat(output.getTextOutput()).contains("test-exec-123");
    }

    @Test
    void descriptionFromTheFlow() throws Exception {
        wireMock.stubFor(get(urlPathMatching("/api/v1/.*/flows/company\\.team/hello-world-with-description"))
            .willReturn(okJson(flowJson("company.team", "hello-world-with-description", 1, "A flow that say Hello World", null))));
        wireMock.stubFor(post(urlPathMatching("/api/v1/.*/executions/company\\.team/hello-world-with-description"))
            .willReturn(okJson(executionJson("test-exec-456", "company.team", "hello-world-with-description"))));

        RunContext runContext = runContextFactory.of(
            Map.of(
                "apiKey", "demo",
                "modelName", "gpt-4o-mini",
                "baseUrl", "http://langchain4j.dev/demo/openai/v1"
            )
        );

        var chat = ChatCompletion.builder()
            .provider(
                OpenAI.builder()
                    .type(OpenAI.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build()
            )
            .tools(
                List.of(
                    KestraFlow.builder()
                        .namespace(Property.ofValue("company.team"))
                        .flowId(Property.ofValue("hello-world-with-description"))
                        .kestraUrl(Property.ofValue("http://localhost:" + wireMock.port()))
                        .build()
                )
            )
            .messages(
                Property.ofValue(
                    List.of(
                        ChatMessage.builder().type(ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                        ChatMessage.builder().type(ChatMessageType.USER).content("I want to execute a flow to say Hello World, please return its response as a valid JSON.").build()
                    )
                )
            )
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(
                ChatConfiguration.builder()
                    .temperature(Property.ofValue(0.1))
                    .seed(Property.ofValue(123456789))
                    .responseFormat(ChatConfiguration.ResponseFormat.builder().type(Property.ofValue(dev.langchain4j.model.chat.request.ResponseFormatType.JSON)).build())
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

        wireMock.verify(postRequestedFor(urlPathMatching("/api/v1/.*/executions/company\\.team/hello-world-with-description")));
    }

    @Test
    void inputsAndLabels() throws Exception {
        String inputsJson = "[{\"id\":\"name\",\"type\":\"STRING\",\"required\":false}]";
        wireMock.stubFor(get(urlPathMatching("/api/v1/.*/flows/company\\.team/hello-world-with-input"))
            .willReturn(okJson(flowJson("company.team", "hello-world-with-input", 1, null, inputsJson))));
        wireMock.stubFor(post(urlPathMatching("/api/v1/.*/executions/company\\.team/hello-world-with-input"))
            .willReturn(okJson(executionJson("test-exec-789", "company.team", "hello-world-with-input"))));

        RunContext runContext = runContextFactory.of(
            Map.of(
                "apiKey", "demo",
                "modelName", "gpt-4o-mini",
                "baseUrl", "http://langchain4j.dev/demo/openai/v1"
            )
        );

        var chat = ChatCompletion.builder()
            .provider(
                OpenAI.builder()
                    .type(OpenAI.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build()
            )
            .tools(
                List.of(
                    KestraFlow.builder()
                        .namespace(Property.ofValue("company.team"))
                        .flowId(Property.ofValue("hello-world-with-input"))
                        .description(Property.ofValue("A flow that say Hello World"))
                        .kestraUrl(Property.ofValue("http://localhost:" + wireMock.port()))
                        .build()
                )
            )
            .messages(
                Property.ofValue(
                    List.of(
                        ChatMessage.builder().type(ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                        ChatMessage.builder().type(ChatMessageType.USER).content("""
                            I want to execute a flow to say Hello World.
                            Call it with the input id 'name' value 'John' and add a label key 'llm' value 'true'.""").build()
                    )
                )
            )
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = chat.run(runContext);
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_flow_company_team_hello-world-with-input");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("kestra_flow_company_team_hello-world-with-input");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();

        wireMock.verify(postRequestedFor(urlPathMatching("/api/v1/.*/executions/company\\.team/hello-world-with-input")));
    }

    @Test
    void helloWorldFromLLM() throws Exception {
        wireMock.stubFor(get(urlPathMatching("/api/v1/.*/flows/company\\.team/hello-world"))
            .willReturn(okJson(flowJson("company.team", "hello-world", 1, "A flow that says Hello World", null))));
        wireMock.stubFor(post(urlPathMatching("/api/v1/.*/executions/company\\.team/hello-world"))
            .willReturn(okJson(executionJson("test-exec-llm", "company.team", "hello-world"))));

        RunContext runContext = runContextFactory.of(
            Map.of(
                "apiKey", "demo",
                "modelName", "gpt-4o-mini",
                "baseUrl", "http://langchain4j.dev/demo/openai/v1"
            )
        );

        var chat = ChatCompletion.builder()
            .provider(
                OpenAI.builder()
                    .type(OpenAI.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build()
            )
            .tools(
                List.of(
                    KestraFlow.builder()
                        .kestraUrl(Property.ofValue("http://localhost:" + wireMock.port()))
                        .build()
                )
            )
            .messages(
                Property.ofValue(
                    List.of(
                        ChatMessage.builder().type(ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                        ChatMessage.builder().type(ChatMessageType.USER)
                            .content("I want to execute the flow 'hello-world' from the namespace 'company.team', please answer with its execution id.").build()
                    )
                )
            )
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

        wireMock.verify(postRequestedFor(urlPathMatching("/api/v1/.*/executions/company\\.team/hello-world")));
        assertThat(output.getTextOutput()).contains("test-exec-llm");
    }
}
