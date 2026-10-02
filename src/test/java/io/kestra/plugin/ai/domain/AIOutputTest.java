package io.kestra.plugin.ai.domain;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.slf4j.Logger;

import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.provider.TimingChatModelListener;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.listener.ChatModelRequestContext;
import dev.langchain4j.model.chat.listener.ChatModelResponseContext;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.service.Result;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@Execution(ExecutionMode.SAME_THREAD)
@ResourceLock("kestra-h2-flyway")
class AIOutputTest {
    private static final ChatRequest REQUEST = ChatRequest.builder()
        .messages(UserMessage.from("hello"))
        .build();

    private final TimingChatModelListener listener = new TimingChatModelListener();
    private final Map<Object, Object> attributes = new HashMap<>();

    @AfterEach
    void tearDown() {
        TimingChatModelListener.clear();
    }

    @Test
    void nullIdToolLoopExtractsFinalDurationBeforeIntermediatesCanConsumeIt() throws Exception {
        RunContext runContext = mock(RunContext.class);
        when(runContext.logger()).thenReturn(mock(Logger.class));

        ChatResponse intermediate = response(null);
        ChatResponse finalResponse = response(null);

        listener.onRequest(new ChatModelRequestContext(REQUEST, null, attributes));
        listener.onResponse(responseContext(intermediate));
        listener.onRequest(new ChatModelRequestContext(REQUEST, null, attributes));
        sleep();
        listener.onResponse(responseContext(finalResponse));

        Result<AiMessage> result = Result.<AiMessage> builder()
            .content(AiMessage.from("done"))
            .tokenUsage(new TokenUsage(2, 1))
            .finishReason(FinishReason.STOP)
            .intermediateResponses(List.of(intermediate))
            .finalResponse(finalResponse)
            .build();

        AIOutput output = AIOutput.from(runContext, result, ResponseFormatType.TEXT);

        assertThat(output.getRequestDuration(), notNullValue());
        assertThat(output.getRequestDuration(), greaterThan(0L));
        assertThat(output.getIntermediateResponses(), hasSize(1));
        assertThat(output.getIntermediateResponses().getFirst().getRequestDuration(), nullValue());
    }

    @Test
    void idPathStillExtractsFromIdKeyedTimers() throws Exception {
        RunContext runContext = mock(RunContext.class);
        when(runContext.logger()).thenReturn(mock(Logger.class));

        ChatResponse finalResponse = response("resp-1");

        listener.onRequest(new ChatModelRequestContext(REQUEST, null, attributes));
        sleep();
        listener.onResponse(responseContext(finalResponse));

        Result<AiMessage> result = Result.<AiMessage> builder()
            .content(AiMessage.from("done"))
            .tokenUsage(new TokenUsage(2, 1))
            .finishReason(FinishReason.STOP)
            .intermediateResponses(List.of())
            .finalResponse(finalResponse)
            .build();

        AIOutput output = AIOutput.from(runContext, result, ResponseFormatType.TEXT);

        assertThat(output.getRequestDuration(), notNullValue());
        assertThat(output.getRequestDuration(), greaterThan(0L));
        assertThat(TimingChatModelListener.pollLastDuration(), nullValue());
    }

    private static ChatResponse response(String id) {
        ChatResponse.Builder builder = ChatResponse.builder()
            .aiMessage(AiMessage.from("chunk"))
            .tokenUsage(new TokenUsage(1, 1));
        if (id != null) {
            builder.id(id);
        }
        return builder.build();
    }

    private ChatModelResponseContext responseContext(ChatResponse chatResponse) {
        return new ChatModelResponseContext(chatResponse, REQUEST, null, attributes);
    }

    private static void sleep() throws InterruptedException {
        Thread.sleep(5);
    }
}
