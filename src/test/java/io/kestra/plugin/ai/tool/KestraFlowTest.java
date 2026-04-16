package io.kestra.plugin.ai.tool;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.sun.net.httpserver.HttpServer;
import dev.langchain4j.model.chat.request.ResponseFormatType;
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
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest(startRunner = true)
class KestraFlowTest {
    @Inject
    private RunContextFactory runContextFactory;

    private HttpServer mockServer;
    private int mockPort;
    private final Map<String, String> stubFlowResponses = new ConcurrentHashMap<>();
    private final Map<String, String> stubExecResponses = new ConcurrentHashMap<>();
    private final AtomicBoolean executionCreated = new AtomicBoolean(false);

    @BeforeEach
    void setUp() throws IOException {
        stubFlowResponses.clear();
        stubExecResponses.clear();
        executionCreated.set(false);
        mockServer = HttpServer.create(new InetSocketAddress(0), 0);
        mockServer.createContext("/", exchange -> {
            String path = exchange.getRequestURI().getPath();
            String method = exchange.getRequestMethod();
            // Drain request body without parsing it as multipart — avoids
            // Apache Commons FileUpload "no multipart boundary" errors
            exchange.getRequestBody().readAllBytes();
            byte[] responseBytes;
            int status;
            if ("GET".equalsIgnoreCase(method)) {
                String body = stubFlowResponses.get(path);
                if (body != null) {
                    responseBytes = body.getBytes(StandardCharsets.UTF_8);
                    status = 200;
                } else {
                    responseBytes = "{}".getBytes(StandardCharsets.UTF_8);
                    status = 404;
                }
            } else if ("POST".equalsIgnoreCase(method)) {
                String body = stubExecResponses.get(path);
                if (body != null) {
                    executionCreated.set(true);
                    responseBytes = body.getBytes(StandardCharsets.UTF_8);
                    status = 200;
                } else {
                    responseBytes = "{}".getBytes(StandardCharsets.UTF_8);
                    status = 404;
                }
            } else {
                responseBytes = "{}".getBytes(StandardCharsets.UTF_8);
                status = 405;
            }
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(status, responseBytes.length);
            try (var os = exchange.getResponseBody()) {
                os.write(responseBytes);
            }
        });
        mockServer.setExecutor(null);
        mockServer.start();
        mockPort = mockServer.getAddress().getPort();
    }

    @AfterEach
    void tearDown() {
        mockServer.stop(0);
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
        // The SDK uses an empty tenant string, producing a double-slash path: /api/v1//flows/...
        stubFlowResponses.put("/api/v1//flows/company.team/hello-world",
            flowJson("company.team", "hello-world", 1, null, null));
        stubExecResponses.put("/api/v1//executions/company.team/hello-world",
            executionJson("test-exec-123", "company.team", "hello-world"));

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
                        .kestraUrl(Property.ofValue("http://localhost:" + mockPort))
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
        assertThat(executionCreated).isTrue();
        assertThat(output.getTextOutput()).contains("test-exec-123");
    }

    @Test
    void descriptionFromTheFlow() throws Exception {
        stubFlowResponses.put("/api/v1//flows/company.team/hello-world-with-description",
            flowJson("company.team", "hello-world-with-description", 1, "A flow that say Hello World", null));
        stubExecResponses.put("/api/v1//executions/company.team/hello-world-with-description",
            executionJson("test-exec-456", "company.team", "hello-world-with-description"));

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
                        .kestraUrl(Property.ofValue("http://localhost:" + mockPort))
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
        assertThat(executionCreated).isTrue();
    }

    @Test
    void inputsAndLabels() throws Exception {
        String inputsJson = "[{\"id\":\"name\",\"type\":\"STRING\",\"required\":false}]";
        stubFlowResponses.put("/api/v1//flows/company.team/hello-world-with-input",
            flowJson("company.team", "hello-world-with-input", 1, null, inputsJson));
        stubExecResponses.put("/api/v1//executions/company.team/hello-world-with-input",
            executionJson("test-exec-789", "company.team", "hello-world-with-input"));

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
                        .kestraUrl(Property.ofValue("http://localhost:" + mockPort))
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
        assertThat(executionCreated).isTrue();
    }

    @Test
    void helloWorldFromLLM() throws Exception {
        stubFlowResponses.put("/api/v1//flows/company.team/hello-world",
            flowJson("company.team", "hello-world", 1, "A flow that says Hello World", null));
        stubExecResponses.put("/api/v1//executions/company.team/hello-world",
            executionJson("test-exec-llm", "company.team", "hello-world"));

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
                        .kestraUrl(Property.ofValue("http://localhost:" + mockPort))
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
        assertThat(executionCreated).isTrue();
        assertThat(output.getTextOutput()).contains("test-exec-llm");
    }
}
