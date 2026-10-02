package io.kestra.plugin.ai.provider;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Configure ordered model providers for fallback",
    description = "Accepts an ordered list of model providers. Fallback invocation behavior is not implemented yet."
)
@Plugin
public class Fallback extends ModelProvider {
    @Schema(
        title = "Providers",
        description = "Model providers in the order they should be considered for fallback."
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<List<ModelProvider>> providers;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        List<ChatModel> childModels = new ArrayList<>();
        for (ModelProvider provider : renderProviders(runContext)) {
            childModels.add(provider.chatModel(runContext, configuration));
        }
        return new FallbackChatModel(childModels, runContext.logger());
    }

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout) throws IllegalVariableEvaluationException {
        List<ChatModel> childModels = new ArrayList<>();
        for (ModelProvider provider : renderProviders(runContext)) {
            childModels.add(provider.chatModel(runContext, configuration, timeout));
        }
        return new FallbackChatModel(childModels, runContext.logger());
    }

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout, List<ChatModelListener> additionalListeners)
        throws IllegalVariableEvaluationException {
        List<ChatModel> childModels = new ArrayList<>();
        for (ModelProvider provider : renderProviders(runContext)) {
            childModels.add(provider.chatModel(runContext, configuration, timeout, additionalListeners));
        }
        return new FallbackChatModel(childModels, runContext.logger());
    }

    private List<ModelProvider> renderProviders(RunContext runContext) throws IllegalVariableEvaluationException {
        return runContext.render(providers).asList(ModelProvider.class);
    }

    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        throw new UnsupportedOperationException("Fallback provider supports chat models only.");
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        throw new UnsupportedOperationException("Fallback provider supports chat models only.");
    }
}
