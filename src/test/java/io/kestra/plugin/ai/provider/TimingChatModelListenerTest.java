package io.kestra.plugin.ai.provider;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.time.StopWatch;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.junit.jupiter.api.parallel.ResourceLock;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.listener.ChatModelRequestContext;
import dev.langchain4j.model.chat.listener.ChatModelResponseContext;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;

@Execution(ExecutionMode.SAME_THREAD)
@ResourceLock("kestra-h2-flyway")
class TimingChatModelListenerTest {
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
    void nullIdResponseSavesDurationThatIsConsumedOnce() throws InterruptedException {
        listener.onRequest(new ChatModelRequestContext(REQUEST, null, attributes));
        Thread.sleep(5);
        listener.onResponse(responseContext(null));

        Long duration = TimingChatModelListener.pollLastDuration();
        assertThat(duration, notNullValue());
        assertThat(duration, greaterThan(0L));
        assertThat(TimingChatModelListener.pollLastDuration(), nullValue());
    }

    @Test
    void pollLastDurationWithoutSavedDurationReturnsNull() {
        TimingChatModelListener.clear();

        assertThat(TimingChatModelListener.pollLastDuration(), nullValue());
    }

    @Test
    void responseWithIdKeepsIdKeyedPathAndLeavesFallbackEmpty() throws InterruptedException {
        listener.onRequest(new ChatModelRequestContext(REQUEST, null, attributes));
        Thread.sleep(5);
        listener.onResponse(responseContext("resp-1"));

        assertThat(TimingChatModelListener.pollLastDuration(), nullValue());
        StopWatch timer = TimingChatModelListener.getTimer("resp-1");
        assertThat(timer, notNullValue());
        assertThat(timer.getTime(TimeUnit.MILLISECONDS), greaterThan(0L));
        assertThat(TimingChatModelListener.getTimer("resp-1"), nullValue());
    }

    @Test
    void clearWipesFallback() {
        listener.onRequest(new ChatModelRequestContext(REQUEST, null, attributes));
        listener.onResponse(responseContext(null));

        TimingChatModelListener.clear();

        assertThat(TimingChatModelListener.pollLastDuration(), nullValue());
    }

    private ChatModelResponseContext responseContext(String responseId) {
        ChatResponse.Builder builder = ChatResponse.builder().aiMessage(AiMessage.from("response"));
        if (responseId != null) {
            builder.id(responseId);
        }
        return new ChatModelResponseContext(builder.build(), REQUEST, null, attributes);
    }
}
