package io.kestra.plugin.ai.memory;

import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.IdUtils;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.rag.ChatCompletion;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@Execution(ExecutionMode.SAME_THREAD)
@KestraTest
class KestraKVStoreTest extends ContainerTest {
    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void testMemory() throws Exception {
        RunContext runContext = runContextFactory.of(
            "namespace", Map.of(
                "modelName", "tinydolphin",
                "embeddingModelName", "chroma/all-minilm-l6-v2-f32",
                "endpoint", ollamaEndpoint,
                "labels", Map.of("system", Map.of("correlationId", IdUtils.create()))
            )
        );

        var rag = ChatCompletion.builder()
            .chatProvider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddingProvider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ embeddingModelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(io.kestra.plugin.ai.embeddings.KestraKVStore.builder().build())
            .memory(KestraKVStore.builder().build())
            .prompt(Property.ofValue("Hello, my name is John"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .chatConfiguration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var ragOutput = rag.run(runContext);
        assertThat(ragOutput.getTextOutput()).isNotNull();

        // call it a second time, it should use the memory
        rag = ChatCompletion.builder()
            .chatProvider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddingProvider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ embeddingModelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .embeddings(io.kestra.plugin.ai.embeddings.KestraKVStore.builder().build())
            .memory(KestraKVStore.builder().build())
            .prompt(Property.ofValue("What's my name?"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .chatConfiguration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        ragOutput = rag.run(runContext);
        assertThat(ragOutput.getTextOutput()).isNotNull();
        assertThat(ragOutput.getTextOutput()).contains("John");
    }
}