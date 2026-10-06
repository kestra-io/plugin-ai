package io.kestra.plugin.ai.provider;

import java.net.ConnectException;
import java.net.SocketTimeoutException;
import java.net.http.HttpConnectTimeoutException;
import java.net.http.HttpTimeoutException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import org.slf4j.Logger;

import dev.langchain4j.exception.JsonException;
import dev.langchain4j.exception.NonRetriableException;
import dev.langchain4j.exception.RetriableException;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.guardrail.GuardrailException;
import dev.langchain4j.model.ModelProvider;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.ChatRequestOptions;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;

public final class FallbackChatModel implements ChatModel {
    private final List<ChatModel> models;
    private final Logger logger;
    private final List<ChatModelListener> listeners;
    private final Set<Capability> supportedCapabilities;

    public FallbackChatModel(List<ChatModel> models, Logger logger) {
        if (models == null || models.isEmpty()) {
            throw new IllegalArgumentException("At least one child chat model is required.");
        }

        this.models = List.copyOf(models);
        this.logger = Objects.requireNonNull(logger, "logger");
        this.listeners = collectListeners(this.models);
        this.supportedCapabilities = collectSupportedCapabilities(this.models);
    }

    @Override
    public ChatResponse chat(ChatRequest request, ChatRequestOptions options) {
        return invoke(model -> model.chat(request, options));
    }

    @Override
    public ChatResponse doChat(ChatRequest request) {
        return invoke(model -> model.doChat(request));
    }

    @Override
    public ChatRequestParameters defaultRequestParameters() {
        // Each child's own chat() call applies that child's defaults. Returning the first
        // model's parameters here preserves useful metadata for consumers that inspect it.
        return models.getFirst().defaultRequestParameters();
    }

    @Override
    public List<ChatModelListener> listeners() {
        // The wrapper dispatches calls to child.chat(), which fires these listeners itself.
        // This list is exposed as metadata but is not fired by the wrapper, avoiding duplicate callbacks.
        return listeners;
    }

    @Override
    public ModelProvider provider() {
        ModelProvider firstProvider = models.getFirst().provider();
        return models.stream().allMatch(model -> model.provider() == firstProvider)
            ? firstProvider
            : ModelProvider.OTHER;
    }

    @Override
    public Set<Capability> supportedCapabilities() {
        return supportedCapabilities;
    }

    private ChatResponse invoke(ChatInvocation invocation) {
        List<Throwable> providerFailures = new ArrayList<>();

        for (int index = 0; index < models.size(); index++) {
            ChatModel model = models.get(index);
            String providerLabel = providerLabel(model);
            logger.info("Attempting chat request with provider {} ({}/{})", providerLabel, index + 1, models.size());

            try {
                return invocation.invoke(model);
            } catch (RuntimeException failure) {
                if (!isProviderFailure(failure)) {
                    logger.warn(
                        "Chat request stopped at provider {} after a non-failover error of type {}",
                        providerLabel,
                        failure.getClass().getSimpleName()
                    );
                    throw failure;
                }

                providerFailures.add(failure);
                logger.warn(
                    "Skipping provider {} after provider-side failure of type {}",
                    providerLabel,
                    failureType(failure)
                );
            }
        }

        RuntimeException finalFailure = (RuntimeException) providerFailures.getLast();
        for (Throwable previousFailure : providerFailures.subList(0, providerFailures.size() - 1)) {
            if (previousFailure != finalFailure) {
                finalFailure.addSuppressed(previousFailure);
            }
        }

        List<String> providerFailureSummaries = new ArrayList<>();
        for (int index = 0; index < providerFailures.size(); index++) {
            providerFailureSummaries.add(
                providerLabel(models.get(index)) + " (" + failureType(providerFailures.get(index)) + ")"
            );
        }
        throw new RuntimeException(
            "All " + providerFailures.size() + " providers failed: " + String.join(", ", providerFailureSummaries),
            finalFailure
        );
    }

    private static boolean isProviderFailure(Throwable failure) {
        List<Throwable> causes = causes(failure);

        // Explicitly reject known application/request failures, even if they have an
        // unexpected retryable exception nested underneath them.
        if (causes.stream().anyMatch(cause ->
            cause instanceof NonRetriableException
                || cause instanceof GuardrailException
                || cause instanceof ToolArgumentsException
                || cause instanceof ToolExecutionException
                || cause instanceof JsonException
                || cause instanceof UnsupportedFeatureException
        )) {
            return false;
        }

        return causes.stream().anyMatch(FallbackChatModel::isEligibleProviderFailureType);
    }

    private static boolean isEligibleProviderFailureType(Throwable failure) {
        return failure instanceof RetriableException
            || failure instanceof ConnectException
            || failure instanceof SocketTimeoutException
            || failure instanceof HttpConnectTimeoutException
            || failure instanceof HttpTimeoutException;
    }

    private static List<Throwable> causes(Throwable failure) {
        List<Throwable> causes = new ArrayList<>();
        Set<Throwable> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        Throwable current = failure;
        while (current != null && seen.add(current)) {
            causes.add(current);
            current = current.getCause();
        }
        return causes;
    }

    private static String failureType(Throwable failure) {
        return causes(failure).stream()
            .filter(FallbackChatModel::isEligibleProviderFailureType)
            .findFirst()
            .map(cause -> cause.getClass().getSimpleName())
            .orElse(failure.getClass().getSimpleName());
    }

    private static String providerName(ChatModel model) {
        ModelProvider provider = model.provider();
        if (provider != null && provider != ModelProvider.OTHER) {
            return provider.name();
        }
        return model.getClass().getSimpleName();
    }

    private static String providerLabel(ChatModel model) {
        String providerName = providerName(model);
        ChatRequestParameters requestParameters = model.defaultRequestParameters();
        String modelName = requestParameters == null ? null : requestParameters.modelName();
        return modelName == null || modelName.isBlank()
            ? providerName
            : providerName + " [model=" + modelName + "]";
    }

    private static List<ChatModelListener> collectListeners(List<ChatModel> models) {
        List<ChatModelListener> allListeners = new ArrayList<>();
        for (ChatModel model : models) {
            for (ChatModelListener listener : model.listeners()) {
                if (allListeners.stream().noneMatch(existing -> existing == listener)) {
                    allListeners.add(listener);
                }
            }
        }
        return List.copyOf(allListeners);
    }

    private static Set<Capability> collectSupportedCapabilities(List<ChatModel> models) {
        Set<Capability> commonCapabilities = new HashSet<>(models.getFirst().supportedCapabilities());
        for (int index = 1; index < models.size(); index++) {
            commonCapabilities.retainAll(models.get(index).supportedCapabilities());
        }
        return Set.copyOf(commonCapabilities);
    }

    @FunctionalInterface
    private interface ChatInvocation {
        ChatResponse invoke(ChatModel model);
    }
}
