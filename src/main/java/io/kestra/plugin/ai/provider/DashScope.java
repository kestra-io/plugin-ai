package io.kestra.plugin.ai.provider;

import java.time.Duration;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;

import dev.langchain4j.community.model.dashscope.*;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import io.kestra.core.models.annotations.PluginProperty;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonDeserialize
@Schema(
    title = "Use DashScope (Qwen) models",
    description = "Calls Alibaba Cloud DashScope for Qwen chat/embeddings/images with API key. Some params (timeouts, retries, stop, maxTokens) map directly to DashScope limits."
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with DashScope (Qwen)",
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
                          type: io.kestra.plugin.ai.provider.DashScope
                          apiKey: "{{ secret('DASHSCOPE_API_KEY') }}"
                          modelName: qwen-plus
                        messages:
                          - type: SYSTEM
                            content: You are a helpful assistant, answer concisely, avoid overly casual language or unnecessary verbosity.
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            }
        )
    }
)
public class DashScope extends ModelProvider {
    private static final String DASHSCOPE_CN_URL = "https://dashscope.aliyuncs.com/api/v1";
    private static final String DASHSCOPE_INTL_URL = "https://dashscope-intl.aliyuncs.com/api/v1";
    private static final String DASHSCOPE_BASE_URL = ZoneId.systemDefault().equals(ZoneId.of("Asia/Shanghai"))
        ? DASHSCOPE_CN_URL
        : DASHSCOPE_INTL_URL;
    @Schema(
        title = "API Key",
        description = "Alibaba Cloud DashScope API key used to authenticate requests. Store it as a Kestra secret rather than inline. No default: this property is required.",
        example = "{{ secret('DASHSCOPE_API_KEY') }}"
    )
    @NotNull
    @PluginProperty(secret = true, group = "main")
    private Property<String> apiKey;

    @Schema(
        title = "API base URL",
        description = "Base URL of the DashScope API. Use `https://dashscope.aliyuncs.com/api/v1` for the China (Beijing) region and `https://dashscope-intl.aliyuncs.com/api/v1` for the Singapore region. Defaults to the region inferred from the worker's system timezone.",
        example = "https://dashscope-intl.aliyuncs.com/api/v1"
    )
    @NotNull
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<String> baseUrl = Property.ofValue(DASHSCOPE_BASE_URL);

    @Schema(
        title = "Repetition penalty",
        description = "Penalty applied to repeated sequences during generation. Higher values reduce repetition; `1.0` means no penalty. Valid range is (0, +inf). Not set by default, in which case the DashScope default applies.",
        example = "1.1"
    )
    @PluginProperty(group = "advanced")
    private Property<Float> repetitionPenalty;

    @Schema(
        title = "Enable Internet search",
        description = "If `true`, the model may use Internet search results as reference when generating text. Defaults to `false`.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    private Property<Boolean> enableSearch;

    @Schema(
        title = "Maximum output tokens",
        description = "Maximum number of tokens returned by a single request. Not set by default, in which case the DashScope default applies.",
        example = "1024"
    )
    @PluginProperty(group = "execution")
    private Property<Integer> maxTokens;

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration)
        throws IllegalVariableEvaluationException {
        return chatModel(runContext, configuration, Duration.ofSeconds(120), Collections.emptyList());
    }

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout, List<ChatModelListener> additionalListeners)
        throws IllegalVariableEvaluationException {
        if (configuration.getLogRequests() != null) {
            throw new IllegalArgumentException(
                "DashScope (Qwen) models do not support setting the logRequests parameter."
            );
        }
        if (configuration.getLogResponses() != null) {
            throw new IllegalArgumentException(
                "DashScope (Qwen) models do not support setting the logResponses parameter."
            );
        }

        var allListeners = new ArrayList<ChatModelListener>();
        allListeners.add(new TimingChatModelListener());
        allListeners.addAll(additionalListeners);

        Double rTemperature = runContext.render(configuration.getTemperature()).as(Double.class).orElse(null);
        Float fTemperature = rTemperature != null ? rTemperature.floatValue() : null;
        return QwenChatModel.builder()
            .baseUrl(runContext.render(baseUrl).as(String.class).orElse(null))
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .modelName(
                runContext.render(this.getModelName()).as(String.class).orElse(QwenModelName.QWEN_PLUS)
            )
            .topP(runContext.render(configuration.getTopP()).as(Double.class).orElse(null))
            .topK(runContext.render(configuration.getTopK()).as(Integer.class).orElse(null))
            .enableSearch(runContext.render(this.enableSearch).as(Boolean.class).orElse(false))
            .seed(runContext.render(configuration.getSeed()).as(Integer.class).orElse(null))
            .repetitionPenalty(runContext.render(this.repetitionPenalty).as(Float.class).orElse(null))
            .temperature(fTemperature)
            .maxTokens(runContext.render(this.maxTokens).as(Integer.class).orElse(null))
            .listeners(allListeners)
            .build();
    }

    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return WanxImageModel.builder()
            .modelName(
                runContext.render(this.getModelName()).as(String.class).orElse(WanxModelName.WANX_V1)
            )
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .baseUrl(runContext.render(baseUrl).as(String.class).orElse(null))
            .build();
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext)
        throws IllegalVariableEvaluationException {
        return QwenEmbeddingModel.builder()
            .modelName(
                runContext
                    .render(this.getModelName())
                    .as(String.class)
                    .orElse(QwenModelName.TEXT_EMBEDDING_V3)
            )
            .apiKey(runContext.render(this.apiKey).as(String.class).orElseThrow())
            .baseUrl(runContext.render(baseUrl).as(String.class).orElse(null))
            .build();
    }
}
