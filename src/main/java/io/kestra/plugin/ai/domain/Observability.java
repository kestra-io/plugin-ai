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
import io.kestra.core.models.annotations.PluginProperty;

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
    @Schema(
        title = "Service name",
        description = "Value reported as the OpenTelemetry `service.name` resource attribute, used to group traces in the observability backend. Defaults to `kestra-plugin-ai`.",
        example = "kestra-plugin-ai"
    )
    @PluginProperty(group = "advanced")
    protected Property<String> serviceName;

    @Schema(
        title = "Environment",
        description = "Deployment environment tagged on the exported traces, so production and staging traffic can be told apart. Not set by default.",
        example = "production"
    )
    @PluginProperty(group = "advanced")
    protected Property<String> environment;

    @Schema(
        title = "Release",
        description = "Application version or release tagged on the exported traces, which helps correlate behavior changes with deployments. Not set by default.",
        example = "1.4.2"
    )
    @PluginProperty(group = "advanced")
    protected Property<String> release;

    @Schema(
        title = "Capture prompt",
        description = "If `true`, prompt content is exported under the span's input attributes. Defaults to `false`, since prompts may carry sensitive data.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    protected Property<Boolean> capturePrompt;

    @Schema(
        title = "Capture system message",
        description = "If `true`, the system message is exported in the span metadata. Defaults to `false`.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    protected Property<Boolean> captureSystemMessage;

    @Schema(
        title = "Capture output",
        description = "If `true`, model output is exported under the span's output attributes. Defaults to `false`, since responses may carry sensitive data.",
        example = "true"
    )
    @PluginProperty(group = "destination")
    protected Property<Boolean> captureOutput;

    @Schema(
        title = "Capture tool arguments",
        description = "If `true`, the arguments passed to each tool are exported in tool execution events. Defaults to `false`.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    protected Property<Boolean> captureToolArguments;

    @Schema(
        title = "Capture tool results",
        description = "If `true`, the value each tool returned is exported in tool execution events. Defaults to `false`.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    protected Property<Boolean> captureToolResults;

    @Schema(
        title = "Maximum payload characters",
        description = "Length cap applied to every captured payload field, beyond which the value is truncated. Defaults to `2000`; a value of `0` or less falls back to that default.",
        example = "2000"
    )
    @PluginProperty(group = "execution")
    protected Property<Integer> maxPayloadChars;

    @Schema(
        title = "Export timeout",
        description = "Time allowed for the OpenTelemetry exporter's flush and shutdown operations before giving up. Defaults to `PT5S`; a zero or negative value falls back to that default.",
        example = "PT5S"
    )
    @PluginProperty(group = "execution")
    protected Property<Duration> exportTimeout;
}
