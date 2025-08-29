package io.kestra.plugin.ai.completion;

import dev.langchain4j.data.image.Image;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.Response;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.AIUtils;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.ai.domain.TokenUsage;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;
import org.slf4j.Logger;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@Schema(
    title = "Generate an image with LLMs",
    description = "Generate images with LLMs using a natural language prompt."
)
@NoArgsConstructor
@Plugin(
    examples = {
        @Example(
            title = "Generate an image using OpenAI (DALL-E 3)",
            full = true,
            code = {
                """
                id: image_generation
                namespace: company.ai

                tasks:
                  - id: image_generation
                    type: io.kestra.plugin.ai.completion.ImageGeneration
                    prompt: >
                      Four-panel comic page about a data engineer shipping a workflow.
                      Clean modern line art with soft colors and ample white space.
                      Panel 1: Early morning desk setup with dual monitors, coffee, and a workflow DAG on screen; calm focused mood.
                      Panel 2: Debugging a failing task; close-up of terminal and error icon; speech bubble: "hmm…"
                      Panel 3: Fix applied; green checks ripple through the pipeline; small celebratory detail (cat paw, fist pump).
                      Panel 4: Deployed dashboard showing metrics trending up; sticky note says "ship it".
                      Include subtle tech props (cloud icons, database cylinder) but no logos.
                      Minimal readable text only in tiny bubbles/notes; no large paragraphs of text.
                    provider:
                      type: io.kestra.plugin.ai.provider.OpenAI
                      apiKey: "{{ kv('OPENAI_API_KEY') }}"
                      modelName: dall-e-3
                """
            }
        ),
    },
    aliases = {"io.kestra.plugin.langchain4j.ImageGeneration", "io.kestra.plugin.langchain4j.completion.ImageGeneration"}
)

public class ImageGeneration extends Task implements RunnableTask<ImageGeneration.Output> {

    @Schema(title = "Image prompt", description = "The input prompt for the image generation model")
    @NotNull
    private Property<String> prompt;

    @Schema(title = "Language Model Provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @Override
    public ImageGeneration.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        // Render input properties
        String renderedPrompt = runContext.render(prompt).as(String.class).orElseThrow();

        // Get the model
        ImageModel model = provider.imageModel(runContext);

        Response<Image> imageUrl = model.generate(renderedPrompt);
        logger.debug("Generated Image URL: {}", imageUrl.content().url());

        // send metrics for token usage
        TokenUsage tokenUsage = TokenUsage.from(imageUrl.tokenUsage());
        AIUtils.sendMetrics(runContext, tokenUsage);

        return Output.builder()
            .imageUrl(String.valueOf(imageUrl.content().url()))
            .tokenUsage(tokenUsage)
            .finishReason(imageUrl.finishReason())
            .build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Generated image URL", description = "The URL of the generated image")
        private String imageUrl;

        @Schema(title = "Token usage")
        private TokenUsage tokenUsage;

        @Schema(title = "Finish reason")
        private FinishReason finishReason;
    }
}