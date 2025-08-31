package io.kestra.plugin.ai.provider;

import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.listener.ChatModelRequestContext;
import dev.langchain4j.model.chat.listener.ChatModelResponseContext;
import org.apache.commons.lang3.time.StopWatch;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class TimingChatModelListener implements ChatModelListener {
    private static final Map<Integer, StopWatch> TIMERS = new HashMap<>();
    private static final Map<String, Integer> TIMER_ID_BY_RESPONSE_ID = new HashMap<>();

    private final AtomicInteger counter = new AtomicInteger(0);

    public static StopWatch getTimer(String responseId) {
        Integer timerId = TIMER_ID_BY_RESPONSE_ID.get(responseId);
        return TIMERS.get(timerId);
    }

    public static void clear() {
        TIMERS.clear();
        TIMER_ID_BY_RESPONSE_ID.clear();
    }

    @Override
    public void onRequest(ChatModelRequestContext requestContext) {
        Integer timerId = counter.incrementAndGet();
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        TIMERS.put(timerId, stopWatch);
        requestContext.attributes().put("kestra.timer.id", timerId);
    }

    @Override
    public void onResponse(ChatModelResponseContext responseContext) {
        Integer timerId = (Integer) responseContext.attributes().get("kestra.timer.id");
        if (timerId == null) {
            return;
        }
        StopWatch stopWatch = TIMERS.get(timerId);
        if (stopWatch == null || !stopWatch.isStarted()) {
            return;
        }
        stopWatch.stop();
        TIMER_ID_BY_RESPONSE_ID.put(responseContext.chatResponse().id(), timerId);
    }
}
