package io.kestra.plugin.ai.domain;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.models.property.Property;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import io.kestra.core.models.annotations.PluginProperty;

@Getter
@SuperBuilder(toBuilder = true)
@NoArgsConstructor
// Concrete subclass must override @JsonDeserialize to avoid StackOverflow with PluginDeserializer.
@JsonDeserialize()
@Schema(
    title = "Langfuse observability",
    description = "OpenTelemetry export settings for Langfuse. Payload capture is disabled by default for security."
)
public class LangfuseObservability extends Observability {
    @Schema(
        title = "Langfuse OTLP endpoint",
        description = "Langfuse OTLP endpoint (for example: https://us.cloud.langfuse.com/api/public/otel)."
    )
    @PluginProperty(group = "connection")
    private Property<String> endpoint;

    @Schema(title = "Langfuse public key")
    @PluginProperty(group = "connection")
    private Property<String> publicKey;

    @Schema(title = "Langfuse secret key")
    @PluginProperty(secret = true, group = "connection")
    private Property<String> secretKey;
}
