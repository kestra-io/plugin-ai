package io.kestra.plugin.ai.domain;

import java.time.Duration;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.plugins.AdditionalPlugin;
import io.kestra.core.plugins.serdes.PluginDeserializer;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Plugin
@SuperBuilder(toBuilder = true)
@Getter
@NoArgsConstructor
// IMPORTANT: The abstract plugin base class must define using the PluginDeserializer,
// AND concrete subclasses must be annotated by @JsonDeserialize() to avoid StackOverflow.
@JsonDeserialize(using = PluginDeserializer.class)
@Schema(
    title = "Observability",
    description = "Observability export settings for AI tasks. Payload capture is disabled by default for security."
)
public abstract class Observability extends AdditionalPlugin {
    @Schema(title = "Enable observability")
    protected Property<Boolean> enabled;

    @Schema(title = "Service name")
    protected Property<String> serviceName;

    @Schema(title = "Environment")
    protected Property<String> environment;

    @Schema(title = "Release")
    protected Property<String> release;

    @Schema(
        title = "Capture prompt",
        description = "If true, prompt content is sent to the observability provider under input attributes. Disabled by default."
    )
    protected Property<Boolean> capturePrompt;

    @Schema(
        title = "Capture system message",
        description = "If true, system message content is sent in metadata. Disabled by default."
    )
    protected Property<Boolean> captureSystemMessage;

    @Schema(
        title = "Capture output",
        description = "If true, model output content is sent to the observability provider under output attributes. Disabled by default."
    )
    protected Property<Boolean> captureOutput;

    @Schema(
        title = "Capture tool arguments",
        description = "If true, tool arguments are sent in tool execution events. Disabled by default."
    )
    protected Property<Boolean> captureToolArguments;

    @Schema(
        title = "Capture tool results",
        description = "If true, tool results are sent in tool execution events. Disabled by default."
    )
    protected Property<Boolean> captureToolResults;

    @Schema(
        title = "Maximum payload characters",
        description = "Maximum number of characters sent for any captured payload field. Longer values are truncated."
    )
    protected Property<Integer> maxPayloadChars;

    @Schema(title = "Export timeout", description = "Timeout used for forceFlush and shutdown operations.")
    protected Property<Duration> exportTimeout;
}
