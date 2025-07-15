package io.kestra.plugin.langchain4j.memory;

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
import io.kestra.plugin.langchain4j.domain.MemoryProvider;
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
    title = "In-memory Chat Memory that then store its serialization form as a Kestra K/V pair",
    description = """
        It will store the memory inside a K/V pair, the name of the entry will be the memory id and it will expires after the memory TTL.
        Be careful that if your internal storage implementation didn't support expiration, the K/V pair may exist forever even if you set a TTL inside the Memory."""
)
@Plugin(
    beta = true,
    examples = {
        @Example(
            full = true,
            title = "Store chat memory inside a K/V pair.",
            code = """
                id: chat-with-memory
                namespace: company.team

                inputs:
                  - id: first
                    type: STRING
                    defaults: Hello, my name is John
                  - id: second
                    type: STRING
                    defaults: What's my name?

                tasks:
                  - id: first
                    type: io.kestra.plugin.langchain4j.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.langchain4j.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddingProvider:
                      type: io.kestra.plugin.langchain4j.provider.GoogleGemini
                      modelName: gemini-embedding-exp-03-07
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.langchain4j.embeddings.KestraKVStore
                    memory:
                      type: io.kestra.plugin.langchain4j.memory.KestraKVMemory
                    systemMessage: You are an helpful assistant, answer concisely
                    prompt: "{{inputs.first}}"
                  - id: second
                    type: io.kestra.plugin.langchain4j.rag.ChatCompletion
                    chatProvider:
                      type: io.kestra.plugin.langchain4j.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddingProvider:
                      type: io.kestra.plugin.langchain4j.provider.GoogleGemini
                      modelName: gemini-embedding-exp-03-07
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.langchain4j.embeddings.KestraKVStore
                    memory:
                      type: io.kestra.plugin.langchain4j.memory.KestraKVMemory
                      drop: true
                    systemMessage: You are an helpful assistant, answer concisely
                    prompt: "{{inputs.second}}"
                """
        ),
    }
)
public class KestraKVMemory extends MemoryProvider {

    @JsonIgnore
    private transient ChatMemory chatMemory;

    @Override
    public ChatMemory chatMemory(RunContext runContext) throws IllegalVariableEvaluationException, IOException {
        this.chatMemory = MessageWindowChatMemory.withMaxMessages(runContext.render(this.getMessages()).as(Integer.class).orElseThrow());

        String key = runContext.render(this.getMemoryId()).as(String.class).orElseThrow();
        Optional<KVEntry> kvEntry = runContext.namespaceKv(runContext.flowInfo().namespace()).get(key);
        if (kvEntry.isPresent() && !kvEntry.get().expirationDate().isBefore(Instant.now())) {
            try {
                KVValue value = runContext.namespaceKv(runContext.flowInfo().namespace()).getValue(kvEntry.get().key()).orElseThrow();
                List<ChatMessage> messages = ChatMessageDeserializer.messagesFromJson((String) value.value());
                messages.forEach(chatMemory::add);
            } catch (ResourceExpiredException ree) {
                // Should not happen as we check for expiry before
                throw new IOException(ree);
            }
        }

        return chatMemory;
    }

    @Override
    public void close(RunContext runContext) throws IllegalVariableEvaluationException, IOException {
        String memoryId = runContext.render(this.getMemoryId()).as(String.class).orElseThrow();
        KVStore kvStore = runContext.namespaceKv(runContext.flowInfo().namespace());
        if (runContext.render(this.getDrop()).as(Boolean.class).orElseThrow()) {
            kvStore.delete(memoryId);
        } else {
            String memoryJson = ChatMessageSerializer.messagesToJson(chatMemory.messages());
            Duration duration = runContext.render(this.getTtl()).as(Duration.class).orElseThrow();
            KVValueAndMetadata kvValueAndMetadata = new KVValueAndMetadata(new KVMetadata("Chat memory for the flow " + runContext.flowInfo().id(), duration), memoryJson);
            kvStore.put(memoryId, kvValueAndMetadata);
        }
    }
}
