package io.kestra.plugin.ai.domain;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;

@Builder
@Getter
public class TokenUsage {
    @Schema(
        title = "Input token count",
        description = "Number of tokens in the prompt sent to the model.",
        example = "320"
    )
    private Integer inputTokenCount;

    @Schema(
        title = "Output token count",
        description = "Number of tokens the model generated in its response.",
        example = "48"
    )
    private Integer outputTokenCount;

    @Schema(
        title = "Total token count",
        description = "Sum of the input and output token counts, which is what the provider bills.",
        example = "368"
    )
    private Integer totalTokenCount;

    public static TokenUsage from(dev.langchain4j.model.output.TokenUsage tokenUsage) {
        if (tokenUsage == null) {
            return null;
        }

        return TokenUsage.builder()
            .inputTokenCount(tokenUsage.inputTokenCount())
            .outputTokenCount(tokenUsage.outputTokenCount())
            .totalTokenCount(tokenUsage.totalTokenCount())
            .build();
    }
}
