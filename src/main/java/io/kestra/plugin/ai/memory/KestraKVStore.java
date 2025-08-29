package io.kestra.plugin.ai.memory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ChatMessageDeserializer;
import dev.langchain4j.data.message.ChatMessageSerializer;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.exceptions.ResourceExpiredException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.runners.RunContext;
import io.kestra.core.storages.kv.*;
import io.kestra.plugin.ai.domain.MemoryProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "In-memory Chat Memory that stores its data as Kestra KV pairs",
    description = """
        Memory stored as a KV pair with key named after the memory id and expiration date defined by the TTL property.
        If your internal storage implementation doesn't support expiration, the KV pair may persist despite the TTL."""
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Store chat memory inside a KV pair",
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
                      type: io.kestra.plugin.ai.memory.KestraKVStore
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
                      type: io.kestra.plugin.ai.memory.KestraKVStore
                      drop: AFTER_TASKRUN
                    systemMessage: You are a helpful assistant, answer concisely
                    prompt: "{{inputs.second}}"
                """
        ),
    },
    aliases = { "io.kestra.plugin.langchain4j.memory.KestraKVMemory", "io.kestra.plugin.ai.memory.KestraKVMemory" }
)
public class KestraKVStore extends MemoryProvider {

    @JsonIgnore
    private transient ChatMemory chatMemory;

    @Override
    public ChatMemory chatMemory(RunContext runContext) throws IllegalVariableEvaluationException, IOException {
        this.chatMemory = MessageWindowChatMemory.withMaxMessages(runContext.render(this.getMessages()).as(Integer.class).orElseThrow());

        String key = runContext.render(this.getMemoryId()).as(String.class).orElseThrow();
        KVStore kvStore = runContext.namespaceKv(runContext.flowInfo().namespace());
        Optional<KVEntry> kvEntry = runContext.namespaceKv(runContext.flowInfo().namespace()).get(key);
        if (kvEntry.isPresent() && !kvEntry.get().expirationDate().isBefore(Instant.now())) {
            try {
                if (runContext.render(this.getDrop()).as(Drop.class).orElse(Drop.NEVER) == Drop.AFTER_TASKRUN) {
                    String rMemoryId = runContext.render(this.getMemoryId()).as(String.class).orElseThrow();
                    kvStore.delete(rMemoryId);
                }
                else {
                    KVValue value = runContext.namespaceKv(runContext.flowInfo().namespace()).getValue(kvEntry.get().key()).orElseThrow();
                    List<ChatMessage> messages = ChatMessageDeserializer.messagesFromJson((String) value.value());
                    messages.forEach(chatMemory::add);
                }
            } catch (ResourceExpiredException ree) {
                // Should not happen as we check for expiry before
                throw new IOException(ree);
            }
        }

        return chatMemory;
    }

    @Override
    public void close(RunContext runContext) throws IllegalVariableEvaluationException, IOException {
        if (chatMemory != null) {
            String rMemoryId = runContext.render(this.getMemoryId()).as(String.class).orElseThrow();
            KVStore kvStore = runContext.namespaceKv(runContext.flowInfo().namespace());
            if (runContext.render(this.getDrop()).as(Drop.class).orElse(Drop.NEVER) == Drop.AFTER_TASKRUN) {
                kvStore.delete(rMemoryId);
            } else {
                String memoryJson = ChatMessageSerializer.messagesToJson(chatMemory.messages());
                Duration duration = runContext.render(this.getTtl()).as(Duration.class).orElseThrow();
                KVValueAndMetadata kvValueAndMetadata = new KVValueAndMetadata(new KVMetadata("Chat memory for the flow " + runContext.flowInfo().id(), duration), memoryJson);
                kvStore.put(rMemoryId, kvValueAndMetadata);
            }
        }
    }
}
