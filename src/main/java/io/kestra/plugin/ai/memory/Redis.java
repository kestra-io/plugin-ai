package io.kestra.plugin.ai.memory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.data.message.ChatMessageDeserializer;
import dev.langchain4j.data.message.ChatMessageSerializer;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.MemoryProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import redis.clients.jedis.Jedis;

import java.io.IOException;
import java.time.Duration;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Persist chat memory in Redis",
    description = """
        Stores chat history in Redis under `memoryId` with TTL-based expiry. Supports drop policies BEFORE/AFTER task run; defaults to KEEP. Ensure Redis is reachable; no TLS/auth fields are defined here."""
)
@Plugin(
    examples = {
        @Example(
            title = "Use Redis-based chat memory for a conversation",
            full = true,
            code = """
                id: chat_with_memory
                namespace: company.ai

                inputs:
                  - id: first
                    type: STRING
                    defaults: Hello, my name is John and I'm from Paris
                  - id: second
                    type: STRING
                    defaults: What's my name and where am I from?

                tasks:
                  - id: first
                    type: io.kestra.plugin.ai.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                    embeddingProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-exp-03-07
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    memory:
                      type: io.kestra.plugin.ai.memory.Redis
                      host: localhost
                      port: 6379
                    systemMessage: You are a helpful assistant, answer concisely
                    prompt: "{{inputs.first}}"

                  - id: second
                    type: io.kestra.plugin.ai.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                    embeddingProvider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-exp-03-07
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    memory:
                      type: io.kestra.plugin.ai.memory.Redis
                      host: localhost
                      port: 6379
                      drop: AFTER_TASKRUN
                    systemMessage: You are a helpful assistant, answer concisely
                    prompt: "{{inputs.second}}"
                """
        )
    }
)
public class Redis extends MemoryProvider {

    @JsonIgnore
    private transient ChatMemory chatMemory;

    @NotNull
    @Schema(
        title = "Redis host",
        description = "The hostname of your Redis server (e.g., localhost or redis-server)"
    )
    private Property<String> host;

    @Schema(
        title = "Redis port",
        description = "The port of your Redis server"
    )
    @Builder.Default
    private Property<Integer> port = Property.ofValue(6379);

    @Override
    public ChatMemory chatMemory(RunContext runContext) throws IllegalVariableEvaluationException, IOException {

        var rHost = runContext.render(this.getHost()).as(String.class).orElseThrow();
        var rPort = runContext.render(this.getPort()).as(Integer.class).orElse(6379);
        var rDrop = runContext.render(this.getDrop()).as(Drop.class).orElse(Drop.NEVER);

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(runContext.render(this.getMessages()).as(Integer.class).orElseThrow());
        var key = runContext.render(this.getMemoryId()).as(String.class).orElseThrow();

        try (var jedis = new Jedis(rHost, rPort)) {
            var json = jedis.get(key);
            if (json != null) {
                if (rDrop == Drop.BEFORE_TASKRUN) {
                    jedis.del(key);
                } else {
                    var messages = ChatMessageDeserializer.messagesFromJson(json);
                    messages.forEach(chatMemory::add);
                }
            }
        }

        return chatMemory;
    }

    @Override
    public void close(RunContext runContext) throws IllegalVariableEvaluationException, IOException {
        if (chatMemory != null) {
            var rHost = runContext.render(this.getHost()).as(String.class).orElseThrow();
            var rPort = runContext.render(this.getPort()).as(Integer.class).orElse(6379);
            var rDrop = runContext.render(this.getDrop()).as(Drop.class).orElse(Drop.NEVER);

            var key = runContext.render(this.getMemoryId()).as(String.class).orElseThrow();

            try (var jedis = new Jedis(rHost, rPort)) {
                if (rDrop == Drop.AFTER_TASKRUN) {
                    jedis.del(key);
                } else {
                    var memoryJson = ChatMessageSerializer.messagesToJson(chatMemory.messages());
                    var ttl = runContext.render(this.getTtl()).as(Duration.class).orElse(Duration.ofMinutes(10));
                    jedis.setex(key, (int) ttl.getSeconds(), memoryJson);
                }
            }
        }
    }
}
