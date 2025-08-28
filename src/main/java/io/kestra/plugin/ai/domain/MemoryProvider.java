package io.kestra.plugin.ai.domain;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.memory.ChatMemory;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.plugins.AdditionalPlugin;
import io.kestra.core.plugins.serdes.PluginDeserializer;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.io.IOException;
import java.time.Duration;

@Plugin
@SuperBuilder(toBuilder = true)
@Getter
@NoArgsConstructor
// IMPORTANT: The abstract plugin base class must define using the PluginDeserializer,
// AND concrete subclasses must be annotated by @JsonDeserialize() to avoid StackOverflow.
@JsonDeserialize(using = PluginDeserializer.class)
public abstract class MemoryProvider extends AdditionalPlugin {
    @Schema(title = "The maximum number of messages to keep inside the memory")
    @Builder.Default
    private Property<Integer> messages = Property.ofValue(10);

    @Schema(title = "The memory duration - defaults to 1h")
    @Builder.Default
    private Property<Duration> ttl = Property.ofValue(Duration.ofHours(1));

    @Schema(title = "The memory id - defaults to the value of the 'system.correlationId' label. This means that a memory is valid for the whole flow execution including its subflows.")
    @Builder.Default
    private Property<String> memoryId = Property.ofExpression("{{ labels.system.correlationId }}");

    @Schema(
        title = "Drop memory: never, before, or after task run",
        description = """
            By default, the memory ID is the value of the 'system.correlationId' label, meaning that the same memory will be used by all tasks of the flow and its subflows.
            If you want to remove the memory eagerly (before expiration), you can set `drop: AFTER_TASKRUN` inside the last task of the flow to erase the memory after its execution.
            You can also set `drop: BEFORE_TASKRUN` to drop the memory before the task execution; this can be useful subflows in case you want a different memory."""
    )
    @Builder.Default
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

    public enum Drop { NEVER, BEFORE_TASKRUN, AFTER_TASKRUN}
}
