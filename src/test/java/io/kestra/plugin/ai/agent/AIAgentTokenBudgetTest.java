package io.kestra.plugin.ai.agent;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextProperty;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.ai.domain.ToolProvider;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.service.tool.ToolExecutor;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AIAgentTokenBudgetTest {
    @Test
    void stopsTheToolLoopWhenTheRenderedTokenBudgetIsExceeded() throws Exception {
        var chatModel = new ScriptedChatModel(
            toolCallResponse(10, 10),
            textResponse(15, 10),
            textResponse(1, 1)
        );
        var tool = new CountingToolProvider();
        var prompt = Property.ofValue("Use the fake tool.");
        var tokenBudget = Property.<Integer> ofExpression("{{ tokenBudget }}");
        RunContext runContext = runContext(prompt, tokenBudget);
        var agent = AIAgent.builder()
            .prompt(prompt)
            .provider(new FakeModelProvider(chatModel))
            .tools(List.of(tool))
            .configuration(
                ChatConfiguration.builder()
                    .maxCumulativeTokens(tokenBudget)
                    .build()
            )
            .build();

        assertThatThrownBy(() -> agent.run(runContext))
            .isInstanceOf(IllegalStateException.class)
            .hasMessage("Cumulative token budget exceeded: consumed 45 tokens, configured `maxCumulativeTokens` is 40.");
        assertThat(chatModel.invocationCount).isEqualTo(2);
        assertThat(tool.invocationCount).hasValue(1);
    }

    @SuppressWarnings({ "rawtypes", "unchecked" })
    private static RunContext runContext(Property<String> prompt, Property<Integer> tokenBudget) throws Exception {
        RunContext runContext = mock(RunContext.class);
        RunContextProperty emptyProperty = mock(RunContextProperty.class);
        RunContextProperty<String> renderedPrompt = mock(RunContextProperty.class);
        RunContextProperty<Integer> renderedBudget = mock(RunContextProperty.class);

        when(runContext.render(nullable(Property.class))).thenReturn(emptyProperty);
        when(runContext.render(prompt)).thenReturn(renderedPrompt);
        when(runContext.render(tokenBudget)).thenReturn(renderedBudget);
        when(emptyProperty.as(any(Class.class))).thenReturn(Optional.empty());
        when(emptyProperty.asList(any(Class.class))).thenReturn(List.of());
        when(renderedPrompt.as(String.class, Map.of())).thenReturn(Optional.of("Use the fake tool."));
        when(renderedBudget.as(Integer.class)).thenReturn(Optional.of(40));
        when(runContext.logger()).thenReturn(LoggerFactory.getLogger(AIAgentTokenBudgetTest.class));
        when(runContext.metric(any())).thenReturn(runContext);

        return runContext;
    }

    private static ChatResponse toolCallResponse(int inputTokens, int outputTokens) {
        var toolExecutionRequest = ToolExecutionRequest.builder()
            .id("call-1")
            .name("fake_tool")
            .arguments("{}")
            .build();

        return ChatResponse.builder()
            .aiMessage(AiMessage.from(toolExecutionRequest))
            .tokenUsage(new TokenUsage(inputTokens, outputTokens))
            .finishReason(FinishReason.TOOL_EXECUTION)
            .build();
    }

    private static ChatResponse textResponse(int inputTokens, int outputTokens) {
        return ChatResponse.builder()
            .aiMessage(AiMessage.from("done"))
            .tokenUsage(new TokenUsage(inputTokens, outputTokens))
            .finishReason(FinishReason.STOP)
            .build();
    }

    private static final class ScriptedChatModel implements ChatModel {
        private final Deque<ChatResponse> responses;
        private int invocationCount;

        private ScriptedChatModel(ChatResponse... responses) {
            this.responses = new ArrayDeque<>(Arrays.asList(responses));
        }

        @Override
        public ChatResponse doChat(ChatRequest request) {
            invocationCount++;
            return responses.removeFirst();
        }
    }

    private static final class FakeModelProvider extends ModelProvider {
        private final ChatModel chatModel;

        private FakeModelProvider(ChatModel chatModel) {
            this.chatModel = chatModel;
        }

        @Override
        public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) {
            return chatModel;
        }

        @Override
        public ImageModel imageModel(RunContext runContext) {
            throw new UnsupportedOperationException();
        }

        @Override
        public EmbeddingModel embeddingModel(RunContext runContext) {
            throw new UnsupportedOperationException();
        }
    }

    private static final class CountingToolProvider extends ToolProvider {
        private final AtomicInteger invocationCount = new AtomicInteger();

        @Override
        public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) {
            var specification = ToolSpecification.builder()
                .name("fake_tool")
                .description("Returns a deterministic result")
                .build();

            return Map.of(specification, (request, memoryId) ->
            {
                invocationCount.incrementAndGet();
                return "tool result";
            });
        }
    }
}
