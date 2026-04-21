package io.kestra.plugin.ai.completion;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import dev.langchain4j.exception.RateLimitException;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.domain.GuardrailRule;
import io.kestra.plugin.ai.domain.Guardrails;
import io.kestra.plugin.ai.provider.GoogleGemini;
import io.kestra.plugin.ai.provider.GoogleVertexAI;
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.provider.OpenAI;
import io.kestra.plugin.ai.provider.OpenRouter;

import jakarta.inject.Inject;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.abort;

@KestraTest
class ClassificationTest extends ContainerTest {
    private final String GEMINI_API_KEY = System.getenv("GEMINI_API_KEY");
    private final String OPENROUTER_API_KEY = System.getenv("OPENROUTER_API_KEY");
    private final String VERTEX_AI_PROJECT = System.getenv("VERTEX_AI_PROJECT");
    private final String VERTEX_AI_LOCATION = System.getenv("VERTEX_AI_LOCATION");

    @Inject
    private RunContextFactory runContextFactory;

    @Test
    @EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
    void testClassificationGemini() throws Exception {
        // GIVEN
        RunContext runContext = runContextFactory.of(
            Map.of(
                "prompt", "Is 'This is a joke' a good joke?",
                "classes", List.of("true", "false"),
                "apiKey", GEMINI_API_KEY,
                "modelName", "gemini-2.0-flash"
            )
        );

        Classification task = Classification.builder()
            .prompt(Property.ofExpression("{{ prompt }}"))
            .systemMessage(Property.ofValue("You are a text classification assistant."))
            .classes(Property.ofExpression("{{ classes }}"))
            .provider(
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .build()
            )
            .build();

        // WHEN / THEN
        try {
            Classification.Output runOutput = task.run(runContext);

            assertThat(runOutput.getClassification(), notNullValue());
        } catch (RateLimitException e) {
            abort("Skipped: Gemini rate limited (429)");
        }
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "VERTEX_AI_PROJECT", matches = ".*")
    @EnabledIfEnvironmentVariable(named = "VERTEX_AI_LOCATION", matches = ".*")
    void testClassificationVertexAI() throws Exception {
        // GIVEN
        RunContext runContext = runContextFactory.of(
            Map.of(
                "prompt", "Is 'This is a joke' a good joke?",
                "classes", List.of("true", "false"),
                "project", VERTEX_AI_PROJECT,
                "location", VERTEX_AI_LOCATION,
                "modelName", "gemini-2.0-flash"
            )
        );

        Classification task = Classification.builder()
            .prompt(Property.ofExpression("{{ prompt }}"))
            .systemMessage(Property.ofValue("You are a text classification assistant."))
            .classes(Property.ofExpression("{{ classes }}"))
            .provider(
                GoogleVertexAI.builder()
                    .type(GoogleVertexAI.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .location(Property.ofExpression("{{ location }}"))
                    .project(Property.ofExpression("{{ project }}"))
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
        RunContext runContext = runContextFactory.of(
            Map.of(
                "classes", List.of("true", "false"),
                "modelName", "tinydolphin",
                "endpoint", ollamaEndpoint
            )
        );

        Classification task = Classification.builder()
            .contentBlocks(Property.ofValue(
                List.of(
                    io.kestra.plugin.ai.domain.ChatMessage.ContentBlock.builder()
                        .text("Is 'This is a joke' a good joke?")
                        .build()
                )
            ))
            .classes(Property.ofExpression("{{ classes }}"))
            .provider(
                Ollama.builder()
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
        RunContext runContext = runContextFactory.of(
            Map.of(
                "prompt", "Is 'This is a joke' a good joke?",
                "classes", List.of("true", "false"),
                "apiKey", "demo",
                "modelName", "gpt-4o-mini",
                "baseUrl", "http://langchain4j.dev/demo/openai/v1"
            )
        );

        Classification task = Classification.builder()
            .prompt(Property.ofExpression("{{ prompt }}"))
            .systemMessage(Property.ofValue("You are a text classification assistant."))
            .classes(Property.ofExpression("{{ classes }}"))
            .provider(
                OpenAI.builder()
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

    @Test
    @EnabledIfEnvironmentVariable(named = "OPENROUTER_API_KEY", matches = ".*")
    void withInputGuardrailViolated() throws Exception {
        RunContext runContext = runContextFactory.of(
            Map.of(
                "apiKey", OPENROUTER_API_KEY,
                "modelName", "openrouter/free",
                "baseUrl", "https://openrouter.ai/api/v1"
            )
        );

        // expression requires prompt shorter than 5 chars — "Hello World" (11 chars) always violates, no LLM call made
        Classification task = Classification.builder()
            .prompt(Property.ofValue("Hello World"))
            .classes(Property.ofValue(List.of("true", "false")))
            .provider(
                OpenRouter.builder()
                    .type(OpenRouter.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build()
            )
            .guardrails(
                Guardrails.builder()
                    .input(
                        List.of(
                            GuardrailRule.builder()
                                .expression("{{ message.length < 5 }}")
                                .message("Prompt too long")
                                .build()
                        )
                    )
                    .build()
            )
            .build();

        Classification.Output output = task.run(runContext);

        assertThat(output.isGuardrailViolated(), is(true));
        assertThat(output.getGuardrailViolationMessage(), containsString("Prompt too long"));
        assertThat(output.getClassification(), nullValue());
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "OPENROUTER_API_KEY", matches = ".*")
    void withOutputGuardrailPasses() throws Exception {
        RunContext runContext = runContextFactory.of(
            Map.of(
                "apiKey", OPENROUTER_API_KEY,
                "modelName", "openrouter/free",
                "baseUrl", "https://openrouter.ai/api/v1"
            )
        );

        Classification task = Classification.builder()
            .prompt(Property.ofValue("Is 'This is a joke' a good joke?"))
            .classes(Property.ofValue(List.of("true", "false")))
            .provider(
                OpenRouter.builder()
                    .type(OpenRouter.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build()
            )
            .guardrails(
                Guardrails.builder()
                    .output(
                        List.of(
                            GuardrailRule.builder()
                                .expression("{{ response.length > 0 }}")
                                .message("Empty response")
                                .build()
                        )
                    )
                    .build()
            )
            .build();

        Classification.Output output = task.run(runContext);

        assertThat(output.isGuardrailViolated(), is(false));
        assertThat(output.getClassification(), notNullValue());
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "OPENROUTER_API_KEY", matches = ".*")
    void withOutputGuardrailViolated() throws Exception {
        RunContext runContext = runContextFactory.of(
            Map.of(
                "apiKey", OPENROUTER_API_KEY,
                "modelName", "openrouter/free",
                "baseUrl", "https://openrouter.ai/api/v1"
            )
        );

        // expression requires response shorter than 1 char — any real LLM response always violates
        Classification task = Classification.builder()
            .prompt(Property.ofValue("Is 'This is a joke' a good joke?"))
            .classes(Property.ofValue(List.of("true", "false")))
            .provider(
                OpenRouter.builder()
                    .type(OpenRouter.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build()
            )
            .guardrails(
                Guardrails.builder()
                    .output(
                        List.of(
                            GuardrailRule.builder()
                                .expression("{{ response.length < 1 }}")
                                .message("Response contains confidential information")
                                .build()
                        )
                    )
                    .build()
            )
            .build();

        Classification.Output output = task.run(runContext);

        assertThat(output.isGuardrailViolated(), is(true));
        assertThat(output.getGuardrailViolationMessage(), containsString("Response contains confidential information"));
        assertThat(output.getClassification(), nullValue());
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "OPENROUTER_API_KEY", matches = ".*")
    void withMultipleGuardrails_firstViolatingRuleWins() throws Exception {
        RunContext runContext = runContextFactory.of(
            Map.of(
                "apiKey", OPENROUTER_API_KEY,
                "modelName", "openrouter/free",
                "baseUrl", "https://openrouter.ai/api/v1"
            )
        );

        // Two input rules: first passes (length < 10000), second fails (length < 5).
        // Output rules are also configured but should never be evaluated — LLM is never called.
        Classification task = Classification.builder()
            .prompt(Property.ofValue("Hello World"))
            .classes(Property.ofValue(List.of("true", "false")))
            .provider(
                OpenRouter.builder()
                    .type(OpenRouter.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build()
            )
            .guardrails(
                Guardrails.builder()
                    .input(
                        List.of(
                            GuardrailRule.builder()
                                .expression("{{ message.length < 10000 }}")
                                .message("Prompt too long")
                                .build(),
                            GuardrailRule.builder()
                                .expression("{{ message.length < 5 }}")
                                .message("Prompt exceeds strict limit")
                                .build()
                        )
                    )
                    .output(
                        List.of(
                            GuardrailRule.builder()
                                .expression("{{ response.length < 1 }}")
                                .message("Should never be reached")
                                .build()
                        )
                    )
                    .build()
            )
            .build();

        Classification.Output output = task.run(runContext);

        assertThat(output.isGuardrailViolated(), is(true));
        assertThat(output.getGuardrailViolationMessage(), containsString("Prompt exceeds strict limit"));
        assertThat(output.getGuardrailViolationMessage(), not(containsString("Should never be reached")));
        assertThat(output.getClassification(), nullValue());
    }

    @Test
    void shouldFailWhenBothPromptAndContentBlocksAreProvided() {
        RunContext runContext = runContextFactory.of(Map.of());

        Classification task = Classification.builder()
            .prompt(Property.ofValue("hello"))
            .contentBlocks(Property.ofValue(
                List.of(
                    io.kestra.plugin.ai.domain.ChatMessage.ContentBlock.builder()
                        .text("hello")
                        .build()
                )
            ))
            .classes(Property.ofValue(List.of("true", "false")))
            .build();

        assertThrows(IllegalArgumentException.class, () -> task.run(runContext));
    }

    @Test
    void shouldFailWhenNeitherPromptNorContentBlocksAreProvided() {
        RunContext runContext = runContextFactory.of(Map.of());

        Classification task = Classification.builder()
            .classes(Property.ofValue(List.of("true", "false")))
            .build();

        assertThrows(IllegalArgumentException.class, () -> task.run(runContext));
    }
}
