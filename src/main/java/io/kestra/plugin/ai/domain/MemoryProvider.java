package io.kestra.plugin.ai.domain;

import java.io.IOException;
import java.time.Duration;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.plugins.AdditionalPlugin;
import io.kestra.core.plugins.serdes.PluginDeserializer;
import io.kestra.core.runners.RunContext;

import dev.langchain4j.memory.ChatMemory;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
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
public abstract class MemoryProvider extends AdditionalPlugin {
    @Schema(
        title = "Maximum messages in memory",
        description = "Number of chat messages retained in memory. When the limit is reached, the oldest messages are evicted first (FIFO); the last system message is always kept. Defaults to `10`.",
        example = "20"
    )
    @Builder.Default
    @PluginProperty(group = "advanced")
    private Property<Integer> messages = Property.ofValue(10);

    @Schema(
        title = "Memory duration",
        description = "How long the memory is retained before it expires. Defaults to `PT1H` (one hour).",
        example = "PT1H"
    )
    @Builder.Default
    @PluginProperty(group = "advanced")
    private Property<Duration> ttl = Property.ofValue(Duration.ofHours(1));

    @Schema(
        title = "Memory ID",
        description = "Identifier under which the conversation is stored, so that different runs or users can keep separate histories. Defaults to the `system.correlationId` label, which makes one memory span an entire flow execution including its subflows.",
        example = "{{ labels.system.correlationId }}"
    )
    @Builder.Default
    @PluginProperty(group = "advanced")
    private Property<String> memoryId = Property.ofExpression("{{ labels.system.correlationId }}");

    @Schema(
        title = "When to drop the memory",
        description = "Controls when the stored conversation is erased, rather than waiting for `ttl` to expire. `NEVER` (default) keeps it until expiry, `BEFORE_TASKRUN` clears it before the task runs, and `AFTER_TASKRUN` clears it once the task run completes.",
        example = "AFTER_TASKRUN"
    )
    @Builder.Default
    @PluginProperty(group = "advanced")
    private Property<Drop> drop = Property.ofValue(Drop.NEVER);

    public abstract ChatMemory chatMemory(RunContext runContext) throws IllegalVariableEvaluationException, IOException;

    /**
     * Tasks to achieve once the operations have been done
     *
     * @param runContext
     * @throws IllegalVariableEvaluationException
     * @throws IOException
     */
    public void close(RunContext runContext) throws IllegalVariableEvaluationException, IOException {
        // by default: no-op
    }

    public enum Drop {
        NEVER,
        BEFORE_TASKRUN,
        AFTER_TASKRUN
    }
}
