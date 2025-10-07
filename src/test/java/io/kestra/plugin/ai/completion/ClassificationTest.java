package io.kestra.plugin.ai.completion;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;
import io.kestra.plugin.ai.provider.GoogleGemini;
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.provider.OpenAI;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.util.List;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;

@KestraTest
class ClassificationTest extends ContainerTest {
    private final String GEMINI_API_KEY = System.getenv("GEMINI_API_KEY");

    @Inject
    private RunContextFactory runContextFactory;

    @Test
    @EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
    void testClassificationGemini() throws Exception {
        // GIVEN
        RunContext runContext = runContextFactory.of(Map.of(
            "messages", List.of(
                ChatMessage.builder()
                    .type(ChatMessageType.SYSTEM)
                    .content("You are a text classification assistant.")
                    .build(),
                ChatMessage.builder()
                    .type(ChatMessageType.USER)
                    .content("Is 'This is a joke' a good joke?")
                    .build()
            ),
            "classes", List.of("true", "false"),
            "apiKey", GEMINI_API_KEY,
            "modelName", "gemini-2.0-flash"
        ));

        Classification task = Classification.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            .classes(Property.ofExpression("{{ classes }}"))
            .provider(GoogleGemini.builder()
                .type(GoogleGemini.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .build()
            )
            .build();

        // WHEN
        Classification.Output runOutput = task.run(runContext);

        // THEN
        assertThat(runOutput.getClassification(), notNullValue());
    }

    @Test
    void testClassificationOllama() throws Exception {
        // GIVEN
        RunContext runContext = runContextFactory.of(Map.of(
            "messages", List.of(
                ChatMessage.builder()
                    .type(ChatMessageType.USER)
                    .content("Is 'This is a joke' a good joke?")
                    .build()
            ),
            "classes", List.of("true", "false"),
            "modelName", "tinydolphin",
            "endpoint", ollamaEndpoint
        ));

        Classification task = Classification.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            .classes(Property.ofExpression("{{ classes }}"))
            .provider(Ollama.builder()
                .type(Ollama.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .endpoint(Property.ofExpression("{{ endpoint }}"))
                .build()
            )
            .build();

        // WHEN
        Classification.Output runOutput = task.run(runContext);

        // THEN
        assertThat(runOutput.getClassification(), notNullValue());
    }

    @Test
    @Disabled("demo apikey has quotas")
    void testClassificationOpenAI() throws Exception {
        // GIVEN
        RunContext runContext = runContextFactory.of(Map.of(
            "messages", List.of(
                ChatMessage.builder()
                    .type(ChatMessageType.SYSTEM)
                    .content("You are a text classification assistant.")
                    .build(),
                ChatMessage.builder()
                    .type(ChatMessageType.USER)
                    .content("Is 'This is a joke' a good joke?")
                    .build()
            ),
            "classes", List.of("true", "false"),
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1"
        ));

        Classification task = Classification.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            .classes(Property.ofExpression("{{ classes }}"))
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .build();

        // WHEN
        Classification.Output runOutput = task.run(runContext);

        // THEN
        assertThat(runOutput.getClassification(), notNullValue());
        assertThat(List.of("true", "false").contains(runOutput.getClassification().toLowerCase()), is(Boolean.TRUE));
    }
}
