package io.kestra.plugin.ai.observability;

import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

import dev.langchain4j.model.ModelProvider;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.ChatRequestOptions;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;

final class ObservedChatModel implements ChatModel {
    private final ChatModel delegate;
    private final TelemetrySession session;
    // LocalAI and Workers AI implement chat(ChatRequest) only. The two-arg default calls doChat, which throws
    // "Not implemented", so those providers have to be invoked through the one-arg method.
    private final boolean legacyChat;

    ObservedChatModel(ChatModel delegate, TelemetrySession session) {
        this.delegate = delegate;
        this.session = session;
        this.legacyChat = legacyChat(delegate);
    }

    @Override
    public ChatResponse chat(ChatRequest chatRequest, ChatRequestOptions options) {
        var span = start(chatRequest);
        if (span == null) {
            return call(chatRequest, options);
        }
        try (Scope ignored = span.makeCurrent()) {
            var response = call(chatRequest, options);
            complete(span, response);
            return response;
        } catch (Exception e) {
            GenAiSpans.fail(session, span, e);
            span = null;
            if (e instanceof RuntimeException runtime) {
                throw runtime;
            }
            throw new IllegalStateException(e);
        } finally {
            GenAiSpans.end(span);
        }
    }

    @Override
    public ChatResponse doChat(ChatRequest chatRequest) {
        return chat(chatRequest);
    }

    @Override
    public CompletableFuture<ChatResponse> chatAsync(ChatRequest chatRequest, ChatRequestOptions options) {
        var span = start(chatRequest);
        if (span == null) {
            return callAsync(chatRequest, options);
        }
        try (Scope ignored = span.makeCurrent()) {
            return callAsync(chatRequest, options).whenComplete((response, error) ->
            {
                if (error != null) {
                    GenAiSpans.fail(session, span, error);
                } else {
                    complete(span, response);
                    GenAiSpans.end(span);
                }
            });
        } catch (Exception e) {
            GenAiSpans.fail(session, span, e);
            throw e instanceof RuntimeException runtime ? runtime : new IllegalStateException(e);
        }
    }

    @Override
    public CompletableFuture<ChatResponse> doChatAsync(ChatRequest chatRequest) {
        return delegate.doChatAsync(chatRequest);
    }

    @Override
    public ChatRequestParameters defaultRequestParameters() {
        return delegate.defaultRequestParameters();
    }

    @Override
    public List<ChatModelListener> listeners() {
        return delegate.listeners();
    }

    @Override
    public ModelProvider provider() {
        return delegate.provider();
    }

    @Override
    public Set<Capability> supportedCapabilities() {
        return delegate.supportedCapabilities();
    }

    private ChatResponse call(ChatRequest chatRequest, ChatRequestOptions options) {
        if (legacyChat) {
            return delegate.chat(chatRequest);
        }
        return delegate.chat(chatRequest, options);
    }

    private CompletableFuture<ChatResponse> callAsync(ChatRequest chatRequest, ChatRequestOptions options) {
        if (legacyChat) {
            return delegate.chatAsync(chatRequest);
        }
        return delegate.chatAsync(chatRequest, options);
    }

    private static boolean legacyChat(ChatModel delegate) {
        var type = delegate.getClass();
        return implemented(type, "chat", ChatRequest.class)
            && !implemented(type, "chat", ChatRequest.class, ChatRequestOptions.class)
            && !implemented(type, "doChat", ChatRequest.class);
    }

    private static boolean implemented(Class<?> type, String name, Class<?>... parameterTypes) {
        try {
            return !type.getMethod(name, parameterTypes).getDeclaringClass().isInterface();
        } catch (NoSuchMethodException e) {
            return false;
        }
    }

    private Span start(ChatRequest chatRequest) {
        Span span = null;
        try {
            var model = GenAiSpans.modelOrFallback(chatRequest.modelName(), session.fallbackModel);
            span = GenAiSpans.start(session, "chat", model);
            if (chatRequest.temperature() != null) {
                span.setAttribute(GenAiSpans.TEMPERATURE, chatRequest.temperature());
            }
            if (chatRequest.topP() != null) {
                span.setAttribute(GenAiSpans.TOP_P, chatRequest.topP());
            }
            GenAiSpans.put(span, GenAiSpans.MAX_TOKENS, chatRequest.maxOutputTokens());
            if (session.captureMessageContent) {
                GenAiSpans.put(span, GenAiSpans.SYSTEM_INSTRUCTIONS, GenAiMessages.systemInstructions(chatRequest.messages()));
                GenAiSpans.put(span, GenAiSpans.INPUT_MESSAGES, GenAiMessages.inputMessages(chatRequest.messages()));
            }
            return span;
        } catch (Exception e) {
            session.logger.debug("Failed to start an LLM span", e);
            GenAiSpans.end(span);
            return null;
        }
    }

    private void complete(Span span, ChatResponse response) {
        try {
            if (response == null) {
                return;
            }
            GenAiSpans.put(span, GenAiSpans.RESPONSE_MODEL, response.modelName());
            GenAiSpans.put(span, GenAiSpans.RESPONSE_ID, response.id());
            if (response.tokenUsage() != null) {
                GenAiSpans.put(span, GenAiSpans.INPUT_TOKENS, response.tokenUsage().inputTokenCount());
                GenAiSpans.put(span, GenAiSpans.OUTPUT_TOKENS, response.tokenUsage().outputTokenCount());
            }
            var finishReason = GenAiMessages.finishReason(response.finishReason());
            if (finishReason != null) {
                span.setAttribute(GenAiSpans.FINISH_REASONS, List.of(finishReason));
            }
            if (session.captureMessageContent && response.aiMessage() != null) {
                GenAiSpans.put(span, GenAiSpans.OUTPUT_MESSAGES, GenAiMessages.outputMessage(response.aiMessage(), response.finishReason()));
            }
        } catch (Exception e) {
            session.logger.debug("Failed to complete an LLM span", e);
        }
    }
}
