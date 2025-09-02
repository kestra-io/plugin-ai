package io.kestra.plugin.ai.tool;

import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.model.output.FinishReason;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.models.tasks.VoidOutput;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.provider.GoogleGemini;
import io.kestra.plugin.ai.provider.OpenAI;
import io.kestra.plugin.core.execution.Fail;
import io.kestra.plugin.core.execution.SetVariables;
import io.kestra.plugin.core.http.Request;
import io.kestra.plugin.core.log.Fetch;
import io.kestra.plugin.core.log.Log;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.inject.Inject;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.Assert.assertThrows;

@KestraTest
class KestraTaskTest extends ContainerTest {
    private final String GEMINI_API_KEY = System.getenv("GEMINI_API_KEY");

    @Inject
    private RunContextFactory runContextFactory;

    @Test
    void logTask() throws Exception {
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
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .tools(List.of(
                KestraTask.builder().tasks(
                    List.of(
                        Log.builder().id("log").type(Log.class.getName()).message("...").build()
                    )
                ).build()
            ))
            .messages(Property.ofValue(
                List.of(
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.USER).content("I want to log the following message: \"Hello World!\"").build()
                )))
            .build();

        var output = chat.run(runContext);
        assertThat(output.getTextOutput()).contains("success");
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_task_log");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("kestra_task_log");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();
    }

    @Test
    void setVariables() throws Exception {
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
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .tools(List.of(
                KestraTask.builder().tasks(
                    List.of(
                        SetVariables.builder().id("setVariables").type(SetVariables.class.getName()).variables(Property.ofExpression("{{agent.variables}}")).build()
                    )
                ).build()
            ))
            .messages(Property.ofValue(
                List.of(
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.USER).content("I want to set a variable of name 'some' and value 'variable'").build()
                )))
            .build();

        IllegalArgumentException exception =  assertThrows(IllegalArgumentException.class, () -> chat.run(runContext));
        assertThat(exception.getMessage()).isEqualTo("The KestraTask tool is only capable of calling runnable tasks but 'setVariables' is not a runnable task.");
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
    void httpRequest() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", GEMINI_API_KEY,
            "modelName", "gemini-2.5-flash"
        ));

        // we use Gemini as the context window of the Langchain4J demo OpenAI API is limited to 5000
        var chat = ChatCompletion.builder()
            .provider(GoogleGemini.builder()
                .type(GoogleGemini.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .build()
            )
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .tools(List.of(
                KestraTask.builder().tasks(
                    List.of(
                        Request.builder().id("request").type(SetVariables.class.getName()).uri(Property.ofExpression("...")).build()
                    )
                ).build()
            ))
            .messages(Property.ofValue(
                List.of(
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.USER).content("""
                            Call the HTTP URL https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/docs/07.architecture/01.main-components.md, retrieve its outputs and extract the body.
                            Add a HTTP Header X-Kestra-API: anything when calling the URL.""")
                        .build()
                )))
            .build();

        var output = chat.run(runContext);
        assertThat(output.getTextOutput()).contains("Technical description of Kestra's main components, including the internal storage, queue, repository, and plugins.");
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_task_request");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("kestra_task_request");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();
    }

    @Test
    void fetchTask() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1",
            "execution", Map.of("id", "executionId")
        ));

        var chat = ChatCompletion.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .tools(List.of(
                KestraTask.builder().tasks(
                    List.of(
                        Fetch.builder().id("fetch").type(Fetch.class.getName()).build()
                    )
                ).build()
            ))
            .messages(Property.ofValue(
                List.of(
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.USER).content("I want to fetch logs from the following task ids: task1, task2 and task3").build()
                )))
            .build();

        var output = chat.run(runContext);
        assertThat(output.getTextOutput()).contains("logs");
        assertThat(output.getTextOutput()).contains("fetch");
        assertThat(output.getTextOutput()).contains("task");
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_task_fetch");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("kestra_task_fetch");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();
    }

    @Test
    void myAwesomeTaskAISetAll() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1",
            "execution", Map.of("id", "executionId")
        ));

        var chat = ChatCompletion.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).logRequests(Property.ofValue(true)).logResponses(Property.ofValue(true)).build())
            .tools(List.of(
                KestraTask.builder().tasks(
                    List.of(
                        MyAwesomeTask.builder().id("awesome").type(MyAwesomeTask.class.getName()).string(Property.ofValue("...")).build()
                    )
                ).build()
            ))
            .messages(Property.ofValue(
                List.of(
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.USER).content("""
                        I want to call my awesome tasks with the following properties:
                        - A number of value 7
                        - A string of value 'Hello World'
                        - An optional with the value 'optional'
                        - A message with key 'mykey' and value 'myvalue'"""
                    ).build()
                )))
            .build();

        var output = chat.run(runContext);
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_task_awesome");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("kestra_task_awesome");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();
        assertThat(output.getTextOutput()).contains("success");
        assertThat(MyAwesomeTask.spy).isEqualTo("7-Hello World-optional-[mykey:myvalue]");
    }

    @Test
    void myAwesomeTaskAISetMandatory() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1",
            "execution", Map.of("id", "executionId")
        ));

        var chat = ChatCompletion.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).logRequests(Property.ofValue(true)).logResponses(Property.ofValue(true)).build())
            .tools(List.of(
                KestraTask.builder().tasks(
                    List.of(
                        MyAwesomeTask.builder().id("awesome").type(MyAwesomeTask.class.getName()).string(Property.ofValue("...")).build()
                    )
                ).build()
            ))
            .messages(Property.ofValue(
                List.of(
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.USER).content("""
                        I want to call my awesome tasks with the following properties:
                        - A number of value 7
                        - A string of value 'Hello World'
                        - A message with key 'mykey' and value 'myvalue'"""
                    ).build()
                )))
            .build();

        var output = chat.run(runContext);
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_task_awesome");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("kestra_task_awesome");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();
        assertThat(output.getTextOutput()).contains("success");
        assertThat(MyAwesomeTask.spy).isEqualTo("7-Hello World-null-[mykey:myvalue]");
    }

    @Test
    void myAwesomeTaskAISetMissingMandatory() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1",
            "execution", Map.of("id", "executionId")
        ));

        var chat = ChatCompletion.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).logRequests(Property.ofValue(true)).logResponses(Property.ofValue(true)).build())
            .tools(List.of(
                KestraTask.builder().tasks(
                    List.of(
                        MyAwesomeTask.builder().id("awesome").type(MyAwesomeTask.class.getName())
                            .string(Property.ofValue("..."))
                            .number(Property.ofValue(8))
                            .build()
                    )
                ).build()
            ))
            .messages(Property.ofValue(
                List.of(
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.USER).content("""
                        I want to call my awesome tasks with the following properties:
                        - A string of value 'Hello World'
                        - A message with key 'mykey' and value 'myvalue'"""
                    ).build()
                )))
            .build();

        var output = chat.run(runContext);
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("kestra_task_awesome");
        assertThat(output.getIntermediateResponses()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests()).isNotEmpty();
        assertThat(output.getIntermediateResponses().getFirst().getToolExecutionRequests().getFirst().getName()).isEqualTo("kestra_task_awesome");
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration()).isNotNull();
        assertThat(output.getTextOutput()).contains("success");
        assertThat(MyAwesomeTask.spy).isEqualTo("8-Hello World-null-[mykey:myvalue]");
    }

    @Test
    void fail() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1",
            "execution", Map.of("id", "executionId")
        ));

        var chat = ChatCompletion.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .tools(List.of(
                KestraTask.builder().tasks(
                    List.of(
                        Fail.builder().id("fail").type(Fail.class.getName()).build()
                    )
                ).build()
            ))
            .messages(Property.ofValue(
                List.of(
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.USER).content("I want to fail").build()
                )))
            .build();

        var exception = assertThrows(ToolExecutionException.class, () -> chat.run(runContext));
        assertThat(exception.getMessage()).isEqualTo("Task failure");
    }

    @SuperBuilder
    @ToString
    @EqualsAndHashCode
    @Getter
    @NoArgsConstructor
    @Plugin
    public static class MyAwesomeTask extends Task implements RunnableTask<VoidOutput> {

        @Schema(title = "A number to set")
        @NotNull
        private Property<@Min(0) Integer> number;

        @Schema(title = "A string to set")
        @NotNull
        private Property<String> string;

        @Schema(title = "An optional property")
        private Property<String> optional;

        @Schema(title = "A message")
        @NotNull
        private Message message;

        static String spy;

        @Override
        public VoidOutput run(RunContext runContext) throws Exception {
            String rNumber = runContext.render(number).as(Integer.class).map(String::valueOf).orElseThrow();
            String rString = runContext.render(string).as(String.class).orElseThrow();
            String rOptional = runContext.render(optional).as(String.class).orElse(null);
            spy = rNumber + "-" + rString + "-" + rOptional + "-[" + message.key + ":" + message.value + "]";
            return null;
        }

        @Schema(title = "A Message")
        @Builder
        @Getter
        @ToString
        public static class Message {
            @Schema(title = "A key")
            @NotNull
            private Object key;

            @Schema(title = "A value")
            @NotNull
            private Object value;
        }
    }
}