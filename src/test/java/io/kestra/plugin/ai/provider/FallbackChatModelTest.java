package io.kestra.plugin.ai.provider;

import java.net.ConnectException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

import org.slf4j.LoggerFactory;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.AuthenticationException;
import dev.langchain4j.exception.InternalServerException;
import dev.langchain4j.exception.InvalidRequestException;
import dev.langchain4j.exception.JsonException;
import dev.langchain4j.exception.ModelNotFoundException;
import dev.langchain4j.exception.NonRetriableException;
import dev.langchain4j.exception.RateLimitException;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.guardrail.InputGuardrailException;
import dev.langchain4j.guardrail.OutputGuardrailException;
import dev.langchain4j.model.ModelProvider;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.ChatRequestOptions;
import dev.langchain4j.model.chat.listener.ChatModelErrorContext;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.listener.ChatModelResponseContext;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class FallbackChatModelTest {
    private static final ChatRequest REQUEST = ChatRequest.builder()
        .messages(UserMessage.from("test prompt"))
        .build();

    @Test
    void orderedSuccessDoesNotCallLaterModels() {
        var firstResponse = response("first response");
        var first = model("first", request -> firstResponse);
        var second = model("second", request -> response("second response"));
        var fallback = fallback(first, second);

        assertThat(fallback.chat(REQUEST)).isSameAs(firstResponse);
        assertThat(first.invocationCount).isEqualTo(1);
        assertThat(second.invocationCount).isZero();
    }

    @Test
    void fallsBackAfterRetriableException() {
        var first = model("first", request -> {
            throw new InternalServerException("provider unavailable");
        });
        var secondResponse = response("second response");
        var second = model("second", request -> secondResponse);

        assertThat(fallback(first, second).chat(REQUEST)).isSameAs(secondResponse);
        assertThat(first.invocationCount).isEqualTo(1);
        assertThat(second.invocationCount).isEqualTo(1);
    }

    @Test
    void fallsBackAfterWrappedConnectionFailure() {
        var first = model("first", request -> {
            // ChatModel does not declare checked exceptions, and the JDK HTTP client
            // wraps connection IO failures in a RuntimeException.
            throw new RuntimeException(new ConnectException("Connection refused"));
        });
        var secondResponse = response("second response");
        var second = model("second", request -> secondResponse);

        assertThat(fallback(first, second).chat(REQUEST)).isSameAs(secondResponse);
        assertThat(second.invocationCount).isEqualTo(1);
    }

    @Test
    void nonRetriableExceptionStopsImmediatelyAndIsPropagated() {
        var failure = new NonRetriableException("do not retry this request");
        var first = model("first", request -> {
            throw failure;
        });
        var second = model("second", request -> response("should not be called"));

        assertThatThrownBy(() -> fallback(first, second).chat(REQUEST))
            .isSameAs(failure);
        assertThat(second.invocationCount).isZero();
    }

    @Test
    void guardrailToolRequestAndOutputFailuresStopImmediately() {
        List<RuntimeException> failures = List.of(
            new InputGuardrailException("input rejected"),
            new OutputGuardrailException("output rejected"),
            new ToolArgumentsException("invalid tool arguments"),
            new ToolExecutionException("tool execution failed"),
            new JsonException("invalid JSON output"),
            new UnsupportedFeatureException("unsupported model feature"),
            new InvalidRequestException("invalid provider request"),
            new AuthenticationException("provider authentication failed"),
            new ModelNotFoundException("model not found"),
            new IllegalArgumentException("unsupported configuration")
        );

        for (RuntimeException failure : failures) {
            var first = model("first", request -> {
                throw failure;
            });
            var second = model("second", request -> response("should not be called"));

            assertThatThrownBy(() -> fallback(first, second).chat(REQUEST))
                .as("failure type %s", failure.getClass().getSimpleName())
                .isSameAs(failure);
            assertThat(second.invocationCount)
                .as("second provider after %s", failure.getClass().getSimpleName())
                .isZero();
        }
    }

    @Test
    void passesTheSameRequestAndOptionsInstancesToEachAttempt() {
        var request = ChatRequest.builder()
            .messages(UserMessage.from("same request"))
            .build();
        var options = ChatRequestOptions.builder()
            .addListenerAttribute("trace", "same options")
            .build();
        var first = model("first", current -> {
            throw new RateLimitException("rate limited");
        });
        var second = model("second", current -> response("success"));

        fallback(first, second).chat(request, options);

        assertThat(first.receivedRequests).containsExactly(request);
        assertThat(second.receivedRequests).containsExactly(request);
        assertThat(first.receivedOptions).containsExactly(options);
        assertThat(second.receivedOptions).containsExactly(options);
    }

    @Test
    void finalProviderFailureContainsEarlierEligibleFailuresAsSuppressed() {
        var firstFailure = new InternalServerException("first unavailable");
        var finalFailure = new RateLimitException("second rate limited");
        var first = model("first", request -> {
            throw firstFailure;
        });
        var second = model("second", request -> {
            throw finalFailure;
        });

        assertThatThrownBy(() -> fallback(first, second).chat(REQUEST))
            .isSameAs(finalFailure)
            .satisfies(thrown -> assertThat(thrown.getSuppressed()).containsExactly(firstFailure));
    }

    @Test
    void childListenersFireOncePerUnderlyingAttempt() {
        var listener = new CountingListener();
        var first = model("first", request -> {
            throw new InternalServerException("first unavailable");
        }, List.of(), Set.of(), DefaultChatRequestParameters.EMPTY, ModelProvider.OPEN_AI, listener);
        var second = model("second", request -> response("success"), List.of(), Set.of(),
            DefaultChatRequestParameters.EMPTY, ModelProvider.GOOGLE_AI_GEMINI, listener);
        var fallback = fallback(first, second);

        assertThat(fallback.listeners()).containsExactly(listener);
        fallback.chat(REQUEST);

        assertThat(listener.errorCount).hasValue(1);
        assertThat(listener.responseCount).hasValue(1);
        assertThat(first.invocationCount).isEqualTo(1);
        assertThat(second.invocationCount).isEqualTo(1);
    }

    @Test
    void advertisesOnlyCapabilitiesCommonToAllModels() {
        var supportsJsonSchema = model("first", request -> response("unused"), List.of(),
            Set.of(Capability.RESPONSE_FORMAT_JSON_SCHEMA), DefaultChatRequestParameters.EMPTY,
            ModelProvider.OPEN_AI);
        var noCapabilities = model("second", request -> response("unused"), List.of(), Set.of(),
            DefaultChatRequestParameters.EMPTY, ModelProvider.GOOGLE_AI_GEMINI);

        assertThat(fallback(supportsJsonSchema, noCapabilities).supportedCapabilities()).isEmpty();

        var alsoSupportsJsonSchema = model("third", request -> response("unused"), List.of(),
            Set.of(Capability.RESPONSE_FORMAT_JSON_SCHEMA), DefaultChatRequestParameters.EMPTY,
            ModelProvider.GOOGLE_AI_GEMINI);
        assertThat(fallback(supportsJsonSchema, alsoSupportsJsonSchema).supportedCapabilities())
            .containsExactly(Capability.RESPONSE_FORMAT_JSON_SCHEMA);
    }

    @Test
    void preservesFirstModelDefaultParametersAndReportsProviderMetadata() {
        ChatRequestParameters firstDefaults = DefaultChatRequestParameters.builder()
            .modelName("first-model")
            .temperature(0.2)
            .build();
        var first = model("first", request -> response("unused"), List.of(), Set.of(),
            firstDefaults, ModelProvider.OPEN_AI);
        var sameProvider = model("second", request -> response("unused"), List.of(), Set.of(),
            DefaultChatRequestParameters.EMPTY, ModelProvider.OPEN_AI);
        var sameProviderFallback = fallback(first, sameProvider);

        assertThat(sameProviderFallback.defaultRequestParameters()).isSameAs(firstDefaults);
        assertThat(sameProviderFallback.provider()).isEqualTo(ModelProvider.OPEN_AI);

        var differentProvider = model("third", request -> response("unused"), List.of(), Set.of(),
            DefaultChatRequestParameters.EMPTY, ModelProvider.GOOGLE_AI_GEMINI);
        assertThat(fallback(first, differentProvider).provider()).isEqualTo(ModelProvider.OTHER);
    }

    @Test
    void genericRuntimeExceptionDoesNotTriggerFallback() {
        var failure = new RuntimeException("application failure");
        var first = model("first", request -> {
            throw failure;
        });
        var second = model("second", request -> response("should not be called"));

        assertThatThrownBy(() -> fallback(first, second).chat(REQUEST))
            .isSameAs(failure);
        assertThat(second.invocationCount).isZero();
    }

    private static FallbackChatModel fallback(TestChatModel... models) {
        return new FallbackChatModel(List.of(models), LoggerFactory.getLogger(FallbackChatModelTest.class));
    }

    private static TestChatModel model(String name, Function<ChatRequest, ChatResponse> behavior) {
        return model(name, behavior, List.of(), Set.of(), DefaultChatRequestParameters.EMPTY, ModelProvider.OTHER);
    }

    private static TestChatModel model(
        String name,
        Function<ChatRequest, ChatResponse> behavior,
        List<ChatModelListener> listeners,
        Set<Capability> capabilities,
        ChatRequestParameters defaultRequestParameters,
        ModelProvider provider,
        ChatModelListener... additionalListeners
    ) {
        List<ChatModelListener> allListeners = new ArrayList<>(listeners);
        allListeners.addAll(List.of(additionalListeners));
        return new TestChatModel(name, behavior, allListeners, capabilities, defaultRequestParameters, provider);
    }

    private static ChatResponse response(String text) {
        return ChatResponse.builder()
            .aiMessage(AiMessage.from(text))
            .build();
    }

    private static final class TestChatModel implements ChatModel {
        private final String name;
        private final Function<ChatRequest, ChatResponse> behavior;
        private final List<ChatModelListener> listeners;
        private final Set<Capability> capabilities;
        private final ChatRequestParameters defaultRequestParameters;
        private final ModelProvider provider;
        private final List<ChatRequest> receivedRequests = new ArrayList<>();
        private final List<ChatRequestOptions> receivedOptions = new ArrayList<>();
        private int invocationCount;

        private TestChatModel(
            String name,
            Function<ChatRequest, ChatResponse> behavior,
            List<ChatModelListener> listeners,
            Set<Capability> capabilities,
            ChatRequestParameters defaultRequestParameters,
            ModelProvider provider
        ) {
            this.name = name;
            this.behavior = behavior;
            this.listeners = listeners;
            this.capabilities = capabilities;
            this.defaultRequestParameters = defaultRequestParameters;
            this.provider = provider;
        }

        @Override
        public ChatResponse chat(ChatRequest request, ChatRequestOptions options) {
            receivedRequests.add(request);
            receivedOptions.add(options);
            return ChatModel.super.chat(request, options);
        }

        @Override
        public ChatResponse doChat(ChatRequest request) {
            invocationCount++;
            return behavior.apply(request);
        }

        @Override
        public ChatRequestParameters defaultRequestParameters() {
            return defaultRequestParameters;
        }

        @Override
        public List<ChatModelListener> listeners() {
            return listeners;
        }

        @Override
        public Set<Capability> supportedCapabilities() {
            return capabilities;
        }

        @Override
        public ModelProvider provider() {
            return provider;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    private static final class CountingListener implements ChatModelListener {
        private final AtomicInteger errorCount = new AtomicInteger();
        private final AtomicInteger responseCount = new AtomicInteger();

        @Override
        public void onError(ChatModelErrorContext errorContext) {
            errorCount.incrementAndGet();
        }

        @Override
        public void onResponse(ChatModelResponseContext responseContext) {
            responseCount.incrementAndGet();
        }
    }
}
