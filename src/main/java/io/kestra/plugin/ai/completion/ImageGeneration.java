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
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.ai.domain.TokenUsage;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;
import org.slf4j.Logger;

import java.util.List;
import java.util.stream.Collectors;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@Schema(
    title = "Generate an image with LLMs",
    description = "Generate images with LLMs using natural language messages."
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
                    messages:
                      - type: SYSTEM
                        content: You are a professional visual designer who creates clean, elegant visuals.
                      - type: USER
                        content: >
                          Four-panel comic page about a data engineer shipping a workflow.
                          Clean modern line art with soft colors and ample white space.
                          Panel 1: Early morning desk setup with dual monitors, coffee, and a workflow DAG on screen.
                          Panel 2: Debugging a failing task; close-up of terminal and error icon.
                          Panel 3: Fix applied; green checks ripple through the pipeline.
                          Panel 4: Deployed dashboard showing metrics trending up; sticky note says "ship it".
                          Include subtle tech props (cloud icons, database cylinder) but no logos.
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

    @Schema(
        title = "Chat Messages",
        description = "The list of chat messages that define the context for the image generation. There can be only one system message, and the last message must be a user message."
    )
    @NotNull
    private Property<List<ChatMessage>> messages;

    @Schema(title = "Image Model Provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @Override
    public ImageGeneration.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        // Render input messages
        List<ChatMessage> rMessages = runContext.render(messages).asList(ChatMessage.class);

        if (rMessages.isEmpty()) {
            throw new IllegalArgumentException("At least one user message must be provided for image generation");
        }

        // Build textual prompt from all user and system messages
        String joinedPrompt = rMessages.stream()
            .map(msg -> switch (msg.type()) {
                case SYSTEM -> "[SYSTEM] " + msg.content();
                case USER -> msg.content();
                case AI -> "[AI] " + msg.content();
            })
            .collect(Collectors.joining("\n"));

        // Get the image model
        ImageModel model = provider.imageModel(runContext);

        // Generate image
        Response<Image> imageResponse = model.generate(joinedPrompt);
        logger.debug("Generated Image URL: {}", imageResponse.content().url());

        // Send token usage metrics
        TokenUsage tokenUsage = TokenUsage.from(imageResponse.tokenUsage());
        AIUtils.sendMetrics(runContext, tokenUsage);

        return Output.builder()
            .imageUrl(String.valueOf(imageResponse.content().url()))
            .tokenUsage(tokenUsage)
            .finishReason(imageResponse.finishReason())
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
