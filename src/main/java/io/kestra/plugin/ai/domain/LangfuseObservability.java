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
        description = "OTLP endpoint traces are exported to, which differs per Langfuse region or self-hosted install. Not set by default, in which case nothing is exported.",
        example = "https://us.cloud.langfuse.com/api/public/otel"
    )
    @PluginProperty(group = "connection")
    private Property<String> endpoint;

    @Schema(
        title = "Langfuse public key",
        description = "Public half of the Langfuse API key pair, sent as the basic-auth username. No default: required for export to work.",
        example = "{{ secret('LANGFUSE_PUBLIC_KEY') }}"
    )
    @PluginProperty(group = "connection")
    private Property<String> publicKey;

    @Schema(
        title = "Langfuse secret key",
        description = "Secret half of the Langfuse API key pair, sent as the basic-auth password. Store it as a Kestra secret rather than inline. No default: required for export to work.",
        example = "{{ secret('LANGFUSE_SECRET_KEY') }}"
    )
    @PluginProperty(secret = true, group = "connection")
    private Property<String> secretKey;
}
