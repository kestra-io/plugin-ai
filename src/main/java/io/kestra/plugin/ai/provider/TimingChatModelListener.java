package io.kestra.plugin.ai.provider;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.lang3.time.StopWatch;

import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.listener.ChatModelRequestContext;
import dev.langchain4j.model.chat.listener.ChatModelResponseContext;

public class TimingChatModelListener implements ChatModelListener {
    private static final Map<Integer, StopWatch> TIMERS = new ConcurrentHashMap<>();
    private static final Map<String, Integer> TIMER_ID_BY_RESPONSE_ID = new ConcurrentHashMap<>();

    // ponytail: static so IDs are globally unique across instances sharing the static maps
    private static final AtomicInteger counter = new AtomicInteger(0);

    public static StopWatch getTimer(String responseId) {
        Integer timerId = TIMER_ID_BY_RESPONSE_ID.remove(responseId);
        return timerId != null ? TIMERS.remove(timerId) : null;
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
        String responseId = responseContext.chatResponse().id();
        if (responseId != null) {
            TIMER_ID_BY_RESPONSE_ID.put(responseId, timerId);
        } else {
            TIMERS.remove(timerId);
        }
    }
}
