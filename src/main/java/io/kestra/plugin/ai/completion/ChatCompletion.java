package io.kestra.plugin.ai.completion;

import dev.langchain4j.data.message.*;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Metric;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.ListUtils;
import io.kestra.plugin.ai.AIUtils;
import io.kestra.plugin.ai.domain.*;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.provider.TimingChatModelListener;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.annotation.Nullable;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;
import org.slf4j.Logger;

import java.util.Collections;
import java.util.List;

import static io.kestra.plugin.ai.domain.ChatMessageType.*;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Run chat completion with tools",
    description = """
        Executes a chat turn from a message list (max one system message; last must be USER) using the configured provider. Optional tools may be invoked by the model; token metrics are reported. JSON-schema formats in `configuration` require provider support."""
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with Google Gemini",
            full = true,
            code = {
                """
                id: chat_completion
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING

                tasks:
                  - id: chat_completion
                    type: io.kestra.plugin.ai.completion.ChatCompletion
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GOOGLE_API_KEY') }}"
                      modelName: gemini-2.5-flash
                    messages:
                      - type: SYSTEM
                        content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                      - type: USER
                        content: "{{inputs.prompt}}"
                """
            }
        ),
        @Example(
            title = "Chat Completion with Google Gemini and a WebSearch tool",
            full = true,
            code = {
                """
                id: chat_completion_with_tools
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING

                tasks:
                  - id: chat_completion_with_tools
                    type: io.kestra.plugin.ai.completion.ChatCompletion
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GOOGLE_API_KEY') }}"
                      modelName: gemini-2.5-flash
                    messages:
                      - type: SYSTEM
                        content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                      - type: USER
                        content: "{{inputs.prompt}}"
                    tools:
                      - type: io.kestra.plugin.ai.tool.GoogleCustomWebSearch
                        apiKey: "{{ secret('GOOGLE_SEARCH_API_KEY') }}"
                        csi: "{{ secret('GOOGLE_SEARCH_CSI') }}"
                """
            }
        ),
        @Example(
            full = true,
            title = """
                Extract structured outputs with a JSON schema.
                Not all model providers support JSON schema; in those cases, you have to specify the schema in the prompt.""",
            code = """
                id: structured-output
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: |
                      Hello, my name is John. I was born on January 1, 2000.

                tasks:
                  - id: ai-agent
                    type: io.kestra.plugin.ai.completion.ChatCompletion
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    configuration:
                      responseFormat:
                        type: JSON
                        jsonSchema:
                          type: object
                          properties:
                            name:
                              type: string
                            birth:
                              type: string
                      messages:
                      - type: USER
                        content: "{{inputs.prompt}}"
                """
        )
    },
    metrics = {
        @Metric(
            name = "input.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) input token count"
        ),
        @Metric(
            name = "output.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) output token count"
        ),
        @Metric(
            name = "total.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) total token count"
        )
    },
    aliases = {"io.kestra.plugin.langchain4j.ChatCompletion", "io.kestra.plugin.langchain4j.completion.ChatCompletion"}
)
public class ChatCompletion extends Task implements RunnableTask<ChatCompletion.Output> {

    @Schema(
        title = "Chat Messages",
        description = "The list of chat messages for the current conversation. There can be only one system message, and the last message must be a user message"
    )
    @NotNull
    protected Property<List<ChatMessage>> messages;

    @Schema(title = "Language Model Provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @Schema(title = "Chat configuration")
    @NotNull
    @PluginProperty
    @Builder.Default
    private ChatConfiguration configuration = ChatConfiguration.empty();

    @Schema(title = "Tools that the LLM may use to augment its response")
    @Nullable
    private List<ToolProvider> tools;

    @Override
    public ChatCompletion.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        // Render existing messages
        List<ChatMessage> renderedChatMessagesInput = runContext.render(messages).asList(ChatMessage.class);
        List<dev.langchain4j.data.message.ChatMessage> chatMessages = convertMessages(renderedChatMessagesInput);

        long nbSystemMessages = chatMessages.stream().filter(msg -> msg.type() == dev.langchain4j.data.message.ChatMessageType.SYSTEM).count();
        if (nbSystemMessages > 1) {
            throw new IllegalArgumentException("Only one system message is allowed");
        }
        if (chatMessages.getLast().type() != dev.langchain4j.data.message.ChatMessageType.USER) {
            throw new IllegalArgumentException("The last message must be a user message");
        }

        // Get the appropriate model from the factory
        ChatModel model = this.provider.chatModel(runContext, configuration);

        List<ToolProvider> toolProviders = ListUtils.emptyOnNull(tools);
        try {
            ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(100); // this should be enough for most use cases
            // add all messages to memory except the system message and the last message that will be used for completion
            List<dev.langchain4j.data.message.ChatMessage> allExceptSystem = chatMessages.stream()
                .filter(msg -> msg.type() != dev.langchain4j.data.message.ChatMessageType.SYSTEM)
                .toList();
            if (allExceptSystem.size() > 1) {
                List<dev.langchain4j.data.message.ChatMessage> history = allExceptSystem.subList(0, allExceptSystem.size() - 1);
                history.forEach(msg -> chatMemory.add(msg));
            }

            // Generate AI response
            Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .systemMessageProvider(chatMemoryId ->
                    chatMessages.stream()
                        .filter(msg -> msg.type() == dev.langchain4j.data.message.ChatMessageType.SYSTEM)
                        .map(msg -> ((SystemMessage)msg).text())
                        .findAny()
                        .orElse(null)
                )
                .chatMemory(chatMemory)
                .tools(AIUtils.buildTools(runContext, Collections.emptyMap(), toolProviders))
                .toolArgumentsErrorHandler((error, context) -> {
                    runContext.logger().error("An error occurred while processing tool arguments for tool {} with request ID {}", context.toolExecutionRequest().name(), context.toolExecutionRequest().id(), error);
                    throw new ToolArgumentsException(error);
                })
                .toolExecutionErrorHandler((error, context) -> {
                    runContext.logger().error("An error occurred during tool execution for tool {} with request ID {}", context.toolExecutionRequest().name(), context.toolExecutionRequest().id(), error);
                    throw new ToolExecutionException(error);
                })
                .build();
            Result<AiMessage> aiResponse = assistant.chat(((UserMessage)chatMessages.getLast()).singleText());
            logger.debug("AI Response: {}", aiResponse.content());

            // send metrics for token usage
            TokenUsage tokenUsage = TokenUsage.from(aiResponse.tokenUsage());
            AIUtils.sendMetrics(runContext, tokenUsage);

            // unfortunately, as we have a deprecated aiResponse field, we have no choice but to first build an AIOutput,
            // then, create the final Output based on it.
            AIOutput output = AIOutput.from(runContext, aiResponse, configuration.computeResponseFormat(runContext).type());
            return Output.builder()
                .aiResponse(output.getTextOutput())
                .tokenUsage(output.getTokenUsage())
                .textOutput(output.getTextOutput())
                .jsonOutput(output.getJsonOutput())
                .finishReason(output.getFinishReason())
                .toolExecutions(output.getToolExecutions())
                .intermediateResponses(output.getIntermediateResponses())
                .requestDuration(output.getRequestDuration())
                .thinking(output.getThinking())
                .sources(output.getSources())
                .build();
        } finally {
            toolProviders.forEach(tool -> tool.close(runContext));

            TimingChatModelListener.clear();
        }
    }

    interface Assistant {
        Result<AiMessage> chat(@dev.langchain4j.service.UserMessage String chatMessage);
    }

    private List<dev.langchain4j.data.message.ChatMessage> convertMessages(List<ChatMessage> messages) {
        return messages.stream()
            .map(dto -> switch (dto.type()) {
                case SYSTEM -> SystemMessage.systemMessage(dto.content());
                case AI ->  AiMessage.aiMessage(dto.content());
                case USER ->  UserMessage.userMessage(dto.content());
            })
            .toList();
    }

    @Override
    public void kill() {
        if (this.tools != null) {
            this.tools.forEach(tool -> {
                try {
                    tool.kill();
                } catch (Exception ignored) {
                }
            });
        }
    }

    @SuperBuilder
    @Getter
    public static class Output extends AIOutput { // we must keep this one to keep the deprecated aiResponse
        @Schema(
            title = "AI Response",
            description = "The generated response from the AI (Deprecated: use `completion` instead)"
        )
        @Deprecated(forRemoval = true, since = "1.0.0")
        private String aiResponse;
    }
}
