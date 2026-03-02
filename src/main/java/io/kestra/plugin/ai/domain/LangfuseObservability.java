package io.kestra.plugin.ai.domain;

import io.kestra.core.models.property.Property;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.Duration;

@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(
    title = "Langfuse observability",
    description = "OpenTelemetry export settings for Langfuse. Payload capture is disabled by default for security."
)
public class LangfuseObservability {
    @Schema(title = "Enable observability")
    @Builder.Default
    private Property<Boolean> enabled = Property.ofValue(false);

    @Schema(
        title = "Langfuse OTLP endpoint",
        description = "Langfuse OTLP endpoint (for example: https://us.cloud.langfuse.com/api/public/otel)."
    )
    private Property<String> endpoint;

    @Schema(title = "Langfuse public key")
    private Property<String> publicKey;

    @Schema(title = "Langfuse secret key")
    private Property<String> secretKey;

    @Schema(title = "Service name")
    @Builder.Default
    private Property<String> serviceName = Property.ofValue("kestra-plugin-ai");

    @Schema(title = "Environment")
    private Property<String> environment;

    @Schema(title = "Release")
    private Property<String> release;

    @Schema(
        title = "Capture prompt",
        description = "If true, prompt content is sent to Langfuse under input attributes. Disabled by default."
    )
    @Builder.Default
    private Property<Boolean> capturePrompt = Property.ofValue(false);

    @Schema(
        title = "Capture system message",
        description = "If true, system message content is sent in metadata. Disabled by default."
    )
    @Builder.Default
    private Property<Boolean> captureSystemMessage = Property.ofValue(false);

    @Schema(
        title = "Capture output",
        description = "If true, model output content is sent to Langfuse under output attributes. Disabled by default."
    )
    @Builder.Default
    private Property<Boolean> captureOutput = Property.ofValue(false);

    @Schema(
        title = "Capture tool arguments",
        description = "If true, tool arguments are sent in tool execution events. Disabled by default."
    )
    @Builder.Default
    private Property<Boolean> captureToolArguments = Property.ofValue(false);

    @Schema(
        title = "Capture tool results",
        description = "If true, tool results are sent in tool execution events. Disabled by default."
    )
    @Builder.Default
    private Property<Boolean> captureToolResults = Property.ofValue(false);

    @Schema(
        title = "Maximum payload characters",
        description = "Maximum number of characters sent for any captured payload field. Longer values are truncated."
    )
    @Builder.Default
    private Property<Integer> maxPayloadChars = Property.ofValue(2000);

    @Schema(title = "Export timeout", description = "Timeout used for forceFlush and shutdown operations.")
    @Builder.Default
    private Property<Duration> exportTimeout = Property.ofValue(Duration.ofSeconds(5));
}
