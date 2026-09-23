package io.kestra.plugin.ai.tool;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.junit.jupiter.api.parallel.ResourceLock;

import com.sun.net.httpserver.HttpServer;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;
import io.kestra.plugin.ai.provider.OpenAI;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.service.tool.ToolExecutor;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@Execution(ExecutionMode.SAME_THREAD)
@ResourceLock("kestra-h2-flyway")
@KestraTest(startRunner = true)
class KestraFlowTest {
    @Inject
    private RunContextFactory runContextFactory;

    private HttpServer mockServer;
    private int mockPort;
    private final Map<String, String> stubFlowResponses = new ConcurrentHashMap<>();
    private final Map<String, String> stubExecResponses = new ConcurrentHashMap<>();
    private final Map<String, Integer> stubStatuses = new ConcurrentHashMap<>();
    private final AtomicBoolean executionCreated = new AtomicBoolean(false);
    private final AtomicInteger requestCount = new AtomicInteger();
    private final AtomicReference<String> lastAuthorizationHeader = new AtomicReference<>();

    @BeforeEach
    void setUp() throws IOException {
        stubFlowResponses.clear();
        stubExecResponses.clear();
        stubStatuses.clear();
        executionCreated.set(false);
        requestCount.set(0);
        lastAuthorizationHeader.set(null);
        mockServer = HttpServer.create(new InetSocketAddress(0), 0);
        mockServer.createContext("/", exchange ->
        {
            String path = exchange.getRequestURI().getPath();
            String method = exchange.getRequestMethod();
            // Drain request body without parsing it as multipart — avoids
            // Apache Commons FileUpload "no multipart boundary" errors
            exchange.getRequestBody().readAllBytes();
            requestCount.incrementAndGet();
            lastAuthorizationHeader.set(exchange.getRequestHeaders().getFirst("Authorization"));
            byte[] responseBytes;
            int status;
            Integer forcedStatus = stubStatuses.get(path);
            if (forcedStatus != null) {
                String errorBody = stubFlowResponses.get(path);
                responseBytes = (errorBody != null ? errorBody : "{}").getBytes(StandardCharsets.UTF_8);
                status = forcedStatus;
            } else if ("GET".equalsIgnoreCase(method)) {
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
        stubFlowResponses.put(
            "/api/v1/main/flows/company.team/hello-world",
            flowJson("company.team", "hello-world", 1, null, null)
        );
        stubExecResponses.put(
            "/api/v1/main/executions/company.team/hello-world",
            executionJson("test-exec-123", "company.team", "hello-world")
        );

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
                        .auth(apiTokenAuth())
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
        stubFlowResponses.put(
            "/api/v1/main/flows/company.team/hello-world-with-description",
            flowJson("company.team", "hello-world-with-description", 1, "A flow that say Hello World", null)
        );
        stubExecResponses.put(
            "/api/v1/main/executions/company.team/hello-world-with-description",
            executionJson("test-exec-456", "company.team", "hello-world-with-description")
        );

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
                        .auth(apiTokenAuth())
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
        stubFlowResponses.put(
            "/api/v1/main/flows/company.team/hello-world-with-input",
            flowJson("company.team", "hello-world-with-input", 1, null, inputsJson)
        );
        stubExecResponses.put(
            "/api/v1/main/executions/company.team/hello-world-with-input",
            executionJson("test-exec-789", "company.team", "hello-world-with-input")
        );

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
                        .auth(apiTokenAuth())
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
        stubFlowResponses.put(
            "/api/v1/main/flows/company.team/hello-world",
            flowJson("company.team", "hello-world", 1, "A flow that says Hello World", null)
        );
        stubExecResponses.put(
            "/api/v1/main/executions/company.team/hello-world",
            executionJson("test-exec-llm", "company.team", "hello-world")
        );

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
                        .auth(apiTokenAuth())
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

    private KestraFlow.Auth apiTokenAuth() {
        return KestraFlow.Auth.builder().apiToken(Property.ofValue("test-token")).build();
    }

    private KestraFlow definedFlowTool(String flowId, KestraFlow.Auth auth) {
        return KestraFlow.builder()
            .namespace(Property.ofValue("company.team"))
            .flowId(Property.ofValue(flowId))
            .description(Property.ofValue("A flow that say Hello World"))
            .kestraUrl(Property.ofValue("http://localhost:" + mockPort))
            .auth(auth)
            .build();
    }

    private ToolExecutor executorOf(KestraFlow tool) throws Exception {
        return tool.tool(runContextFactory.of(), Map.of()).values().iterator().next();
    }

    private KestraFlow.AllowedFlow allowedFlow(String namespace, String flowId) {
        return KestraFlow.AllowedFlow.builder()
            .namespace(namespace == null ? null : Property.ofValue(namespace))
            .flowId(flowId == null ? null : Property.ofValue(flowId))
            .build();
    }

    private KestraFlow genericFlowTool(List<KestraFlow.AllowedFlow> allowedFlows) {
        return KestraFlow.builder()
            .kestraUrl(Property.ofValue("http://localhost:" + mockPort))
            .auth(apiTokenAuth())
            .allowedFlows(allowedFlows)
            .build();
    }

    private void stubSuccessfulFlow(String namespace, String flowId) {
        stubFlowResponses.put(
            "/api/v1/main/flows/" + namespace + "/" + flowId,
            flowJson(namespace, flowId, 1, "An allowed flow", null)
        );
        stubExecResponses.put(
            "/api/v1/main/executions/" + namespace + "/" + flowId,
            executionJson("test-exec-123", namespace, flowId)
        );
    }

    private ToolExecutionRequest flowRequest(String arguments) {
        return ToolExecutionRequest.builder().id("1").name("kestra_flow").arguments(arguments).build();
    }

    @Test
    void shouldDescribeGenericFlowParametersWhenAllowlistIsOmitted() throws Exception {
        var specification = genericFlowTool(null).tool(runContextFactory.of(), Map.of()).keySet().iterator().next();

        assertThat(specification.parameters().required()).contains("namespace", "flowId");
        assertThat(specification.parameters().properties().get("namespace"))
            .isInstanceOfSatisfying(JsonStringSchema.class, schema -> assertThat(schema.description()).contains("existing Kestra flow"));
        assertThat(specification.parameters().properties().get("flowId"))
            .isInstanceOfSatisfying(JsonStringSchema.class, schema -> assertThat(schema.description()).contains("selected namespace"));
        assertThat(requestCount).hasValue(0);
    }

    @Test
    void shouldExposeRenderedAllowedFlowPairsFromYaml() throws Exception {
        var tool = JacksonMapper.ofYaml().readValue("""
            type: io.kestra.plugin.ai.tool.KestraFlow
            kestraUrl: http://localhost:%d
            auth:
              auto: false
            allowedFlows:
              - namespace: "{{ allowedNamespace }}"
                flowId: hello-world
              - namespace: company.other
                flowId: "{{ allowedFlowId }}"
              - namespace: company.team
                flowId: goodbye
            """.formatted(mockPort), KestraFlow.class);
        var tools = tool.tool(runContextFactory.of(), Map.of("allowedNamespace", "company.team", "allowedFlowId", "goodbye"));
        var specification = tools.keySet().iterator().next();

        assertThat(specification.description())
            .contains("(company.team, hello-world)", "(company.other, goodbye)", "(company.team, goodbye)")
            .contains("Do not combine values from different pairs");
        assertThat(specification.parameters().properties().get("namespace"))
            .isInstanceOfSatisfying(JsonEnumSchema.class, schema ->
            {
                assertThat(schema.enumValues()).containsExactly("company.team", "company.other");
                assertThat(schema.description()).contains("existing Kestra flow");
            });
        assertThat(specification.parameters().properties().get("flowId"))
            .isInstanceOfSatisfying(JsonEnumSchema.class, schema -> assertThat(schema.enumValues()).containsExactly("hello-world", "goodbye"));

        stubSuccessfulFlow("company.other", "goodbye");
        var result = tools.values().iterator().next().execute(flowRequest("{\"namespace\":\"company.other\",\"flowId\":\"goodbye\"}"), "memory");
        assertThat(JacksonMapper.toMap(result)).containsEntry("id", "test-exec-123").containsEntry("flowId", "goodbye");
        assertThat(executionCreated).isTrue();
        assertThat(requestCount).hasValue(2);
    }

    @Test
    void shouldRejectDisallowedAndCrossPairedFlowsBeforeCallingTheApi() throws Exception {
        var tool = genericFlowTool(List.of(allowedFlow("company.team", "hello-world"), allowedFlow("company.other", "goodbye")));
        var executor = executorOf(tool);
        var rejectedArguments = List.of(
            "{\"namespace\":\"company.team\",\"flowId\":\"unknown\"}",
            "{\"namespace\":\"unknown\",\"flowId\":\"hello-world\"}",
            "{\"namespace\":\"company.team\",\"flowId\":\"goodbye\"}",
            "{\"namespace\":\"company.other\",\"flowId\":\"hello-world\"}",
            "{\"namespace\":\"company.team\"}",
            "{\"namespace\":\"company.team\",\"flowId\":\"\"}"
        );

        for (var arguments : rejectedArguments) {
            assertThatThrownBy(() -> executor.execute(flowRequest(arguments), "memory"))
                .isInstanceOf(ToolArgumentsException.class)
                .hasMessageContaining("not in allowedFlows")
                .hasMessageContaining("Select an exact namespace and flowId pair");
        }
        assertThat(executionCreated).isFalse();
        assertThat(requestCount).hasValue(0);
    }

    @Test
    void shouldExecuteGenericFlowWhenAllowlistIsOmitted() throws Exception {
        stubSuccessfulFlow("company.team", "hello-world");

        var result = executorOf(genericFlowTool(null)).execute(flowRequest("{\"namespace\":\"company.team\",\"flowId\":\"hello-world\"}"), "memory");

        assertThat(JacksonMapper.toMap(result)).containsEntry("id", "test-exec-123");
        assertThat(executionCreated).isTrue();
        assertThat(requestCount).hasValue(2);
    }

    @Test
    void shouldRejectEmptyAllowlistBeforeCallingTheApi() {
        assertThatThrownBy(() -> executorOf(genericFlowTool(List.of())))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("allowedFlows must contain at least one flow");
        assertThat(requestCount).hasValue(0);
    }

    @Test
    void shouldRejectNullAllowlistEntryBeforeCallingTheApi() {
        assertThatThrownBy(() -> executorOf(genericFlowTool(Collections.singletonList(null))))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("null entries are not allowed");
        assertThat(requestCount).hasValue(0);
    }

    @Test
    void shouldRejectMissingOrBlankAllowedFlowIdentifiers() {
        for (var invalid : Arrays.asList(null, "", " \t")) {
            assertThatThrownBy(() -> executorOf(genericFlowTool(List.of(allowedFlow(invalid, "hello-world")))))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("nonblank namespace");
            assertThatThrownBy(() -> executorOf(genericFlowTool(List.of(allowedFlow("company.team", invalid)))))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("nonblank flowId");
        }
        assertThat(requestCount).hasValue(0);
    }

    @Test
    void shouldRejectAllowedFlowExpressionThatRendersBlank() {
        var tool = genericFlowTool(
            List.of(
                KestraFlow.AllowedFlow.builder()
                    .namespace(Property.ofExpression("{{ allowedNamespace }}"))
                    .flowId(Property.ofValue("hello-world"))
                    .build()
            )
        );

        assertThatThrownBy(() -> tool.tool(runContextFactory.of(), Map.of("allowedNamespace", " ")))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("nonblank namespace");
        assertThat(requestCount).hasValue(0);
    }

    @Test
    void shouldEnforceAllowlistForPredefinedFlowsBeforeCallingTheApi() {
        var tool = KestraFlow.builder()
            .namespace(Property.ofValue("company.team"))
            .flowId(Property.ofValue("hello-world"))
            .allowedFlows(List.of(allowedFlow("company.other", "hello-world")))
            .kestraUrl(Property.ofValue("http://localhost:" + mockPort))
            .auth(apiTokenAuth())
            .build();

        assertThatThrownBy(() -> executorOf(tool))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("predefined flow 'hello-world'")
            .hasMessageContaining("not in allowedFlows");
        assertThat(requestCount).hasValue(0);
    }

    @Test
    void shouldExecuteAllowedPredefinedFlowWithoutAcceptingModelTargetOverrides() throws Exception {
        stubSuccessfulFlow("company.team", "hello-world");
        var tool = KestraFlow.builder()
            .namespace(Property.ofValue("company.team"))
            .flowId(Property.ofValue("hello-world"))
            .allowedFlows(List.of(allowedFlow("company.team", "hello-world")))
            .kestraUrl(Property.ofValue("http://localhost:" + mockPort))
            .auth(apiTokenAuth())
            .build();
        var tools = tool.tool(runContextFactory.of(), Map.of());
        assertThat(tools.keySet().iterator().next().parameters().properties()).doesNotContainKeys("namespace", "flowId");

        var result = tools.values().iterator().next().execute(flowRequest("{\"namespace\":\"other\",\"flowId\":\"denied\"}"), "memory");

        assertThat(JacksonMapper.toMap(result)).containsEntry("id", "test-exec-123").containsEntry("flowId", "hello-world");
        assertThat(executionCreated).isTrue();
        assertThat(requestCount).hasValue(2);
    }

    @Test
    void shouldExecutePredefinedFlowWhenAllowlistIsOmitted() throws Exception {
        stubSuccessfulFlow("company.team", "hello-world");

        var result = executorOf(definedFlowTool("hello-world", apiTokenAuth())).execute(flowRequest("{}"), "memory");

        assertThat(JacksonMapper.toMap(result)).containsEntry("id", "test-exec-123");
        assertThat(executionCreated).isTrue();
        assertThat(requestCount).hasValue(2);
    }

    @Test
    void shouldReportAnAuthenticationFailureWhenTheApiRejectsTheCredentials() {
        stubStatuses.put("/api/v1/main/flows/company.team/hello-world", 401);

        var tool = definedFlowTool("hello-world", apiTokenAuth());

        assertThatThrownBy(() -> tool.tool(runContextFactory.of(), Map.of()))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Authentication failed")
            .hasMessageContaining("hello-world");
    }

    @Test
    void shouldReportAnAuthorizationFailureWhenTheApiForbidsTheFlow() {
        stubStatuses.put("/api/v1/main/flows/company.team/hello-world", 403);

        var tool = definedFlowTool("hello-world", apiTokenAuth());

        assertThatThrownBy(() -> tool.tool(runContextFactory.of(), Map.of()))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Not authorized");
    }

    @Test
    void shouldStillReportAMissingFlowWhenTheApiReturnsNotFound() {
        var tool = definedFlowTool("unknown-flow", apiTokenAuth());

        assertThatThrownBy(() -> tool.tool(runContextFactory.of(), Map.of()))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Unable to find the flow 'unknown-flow'");
    }

    @Test
    void shouldReportAnAuthenticationFailureWhenTheFlowIsResolvedByTheLlm() throws Exception {
        stubStatuses.put("/api/v1/main/flows/company.team/hello-world", 401);

        var tool = KestraFlow.builder().kestraUrl(Property.ofValue("http://localhost:" + mockPort)).auth(apiTokenAuth()).build();
        var executor = executorOf(tool);
        var request = ToolExecutionRequest.builder()
            .id("1")
            .name("kestra_flow")
            .arguments("{\"namespace\":\"company.team\",\"flowId\":\"hello-world\"}")
            .build();

        assertThatThrownBy(() -> executor.execute(request, "memory"))
            .isInstanceOf(ToolExecutionException.class)
            .hasMessageStartingWith("Authentication failed")
            .cause().hasMessageStartingWith("Authentication failed");
    }

    @Test
    void shouldReportAnAuthenticationFailureWhenTheExecutionIsRejected() throws Exception {
        stubFlowResponses.put(
            "/api/v1/main/flows/company.team/hello-world",
            flowJson("company.team", "hello-world", 1, null, null)
        );
        stubStatuses.put("/api/v1/main/executions/company.team/hello-world", 401);

        var tool = definedFlowTool("hello-world", apiTokenAuth());
        var executor = executorOf(tool);
        var request = ToolExecutionRequest.builder().id("1").name("kestra_flow").arguments("{}").build();

        assertThatThrownBy(() -> executor.execute(request, "memory"))
            .isInstanceOf(ToolExecutionException.class)
            .hasMessageStartingWith("Authentication failed")
            .cause().hasMessageStartingWith("Authentication failed");
    }

    @Test
    void shouldFailBeforeCallingTheApiWhenNoCredentialsCanBeRetrieved() {
        var tool = definedFlowTool("hello-world", null);

        assertThatThrownBy(() -> tool.tool(runContextFactory.of(), Map.of()))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("No authentication method provided");
        assertThat(requestCount).hasValue(0);
    }

    @Test
    void shouldReportAMissingInputAsAnArgumentsProblem() throws Exception {
        stubFlowResponses.put(
            "/api/v1/main/flows/company.team/hello-world",
            flowJson("company.team", "hello-world", 1, null, "[{\"id\":\"name\",\"type\":\"STRING\",\"required\":true}]")
        );

        var tool = definedFlowTool("hello-world", apiTokenAuth());
        var executor = executorOf(tool);
        var request = ToolExecutionRequest.builder().id("1").name("kestra_flow").arguments("{}").build();

        assertThatThrownBy(() -> executor.execute(request, "memory"))
            .isInstanceOf(ToolArgumentsException.class)
            .hasMessageContaining("'name'");
    }

    /**
     * Opting out of the default authentication without setting any credential is how the tool declares that the
     * Kestra API it targets requires none, so the call must go out with no `Authorization` header at all.
     */
    @Test
    void shouldSendNoAuthorizationHeaderWhenAutoIsDisabledWithoutCredentials() throws Exception {
        stubFlowResponses.put(
            "/api/v1/main/flows/company.team/hello-world",
            flowJson("company.team", "hello-world", 1, null, null)
        );

        var tool = definedFlowTool("hello-world", KestraFlow.Auth.builder().auto(Property.ofValue(false)).build());

        tool.tool(runContextFactory.of(), Map.of());

        assertThat(requestCount).hasValue(1);
        assertThat(lastAuthorizationHeader).hasNullValue();
    }

    /** Counterpart of the test above: a credential still reaches the API, so an absent header there means something. */
    @Test
    void shouldSendTheApiTokenAsABearerHeader() throws Exception {
        stubFlowResponses.put(
            "/api/v1/main/flows/company.team/hello-world",
            flowJson("company.team", "hello-world", 1, null, null)
        );

        var tool = definedFlowTool("hello-world", apiTokenAuth());

        tool.tool(runContextFactory.of(), Map.of());

        assertThat(lastAuthorizationHeader).hasValue("Bearer test-token");
    }

    @Test
    void shouldIncludeTheApiMessageWhenTheStatusIsNotMapped() {
        stubStatuses.put("/api/v1/main/flows/company.team/hello-world", 500);
        stubFlowResponses.put(
            "/api/v1/main/flows/company.team/hello-world",
            "{\"message\":\"the database is unreachable\"}"
        );

        var tool = definedFlowTool("hello-world", apiTokenAuth());

        assertThatThrownBy(() -> tool.tool(runContextFactory.of(), Map.of()))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("returned the status 500")
            .hasMessageContaining("the database is unreachable");
    }

    @Test
    void shouldReportAnUnreachableApiInsteadOfAMissingFlow() {
        var tool = KestraFlow.builder()
            .namespace(Property.ofValue("company.team"))
            .flowId(Property.ofValue("hello-world"))
            .description(Property.ofValue("A flow that say Hello World"))
            .kestraUrl(Property.ofValue("http://localhost:1"))
            .auth(apiTokenAuth())
            .build();

        assertThatThrownBy(() -> tool.tool(runContextFactory.of(), Map.of()))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("could not be reached");
    }
}
