package io.kestra.plugin.ai.completion;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import com.fasterxml.jackson.databind.JsonNode;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.domain.GuardrailRule;
import io.kestra.plugin.ai.domain.Guardrails;
import io.kestra.plugin.ai.provider.GoogleGemini;
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

@KestraTest
class JSONStructuredExtractionTest extends ContainerTest {
    private final String GEMINI_API_KEY = System.getenv("GEMINI_API_KEY");
    private final String OPENROUTER_API_KEY = "";//System.getenv("OPENROUTER_API_KEY");

    @Inject
    private RunContextFactory runContextFactory;

    @Test
    @EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
    void testJSONStructuredGemini() throws Exception {
        // GIVEN
        RunContext runContext = runContextFactory.of(
            Map.of(
                "prompt", "Hello, my name is John. I was born on January 1, 2000.",
                "systemMessage", "You extract structured JSON data from natural language text.",
                "jsonFields", List.of("name", "date"),
                "schemaName", "Person",
                "modelName", "gemini-2.0-flash",
                "apiKey", GEMINI_API_KEY
            )
        );

        JSONStructuredExtraction task = JSONStructuredExtraction.builder()
            .prompt(Property.ofExpression("{{ prompt }}"))
            .systemMessage(Property.ofExpression("{{ systemMessage }}"))
            .schemaName(Property.ofExpression("{{ schemaName }}"))
            .jsonFields(Property.ofExpression("{{ jsonFields }}"))
            .provider(
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .build()
            )
            .build();

        // WHEN
        JSONStructuredExtraction.Output runOutput = task.run(runContext);

        // THEN
        assertThat(runOutput.getExtractedJson(), notNullValue());

        JsonNode json = JacksonMapper.ofJson().readTree(runOutput.getExtractedJson());
        assertThat(json.has("name"), is(true));
        assertThat(json.has("date"), is(true));
    }

    @Test
    void testJSONStructuredOllama() throws Exception {
        // GIVEN
        RunContext runContext = runContextFactory.of(
            Map.of(
                "prompt", "Hello, my name is Alice, I live in London.",
                "schemaName", "Person",
                "jsonFields", List.of("name", "city"),
                "modelName", "tinydolphin",
                "endpoint", ollamaEndpoint
            )
        );

        JSONStructuredExtraction task = JSONStructuredExtraction.builder()
            .prompt(Property.ofExpression("{{ prompt }}"))
            .schemaName(Property.ofExpression("{{ schemaName }}"))
            .jsonFields(Property.ofExpression("{{ jsonFields }}"))
            .provider(
                Ollama.builder()
                    .type(Ollama.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .endpoint(Property.ofExpression("{{ endpoint }}"))
                    .build()
            )
            .build();

        // WHEN
        JSONStructuredExtraction.Output runOutput = task.run(runContext);

        // THEN
        assertThat(runOutput.getExtractedJson(), notNullValue());
        assertThat(runOutput.getExtractedJson().toLowerCase().contains("alice"), is(Boolean.TRUE));
        assertThat(runOutput.getExtractedJson().toLowerCase().contains("london"), is(Boolean.TRUE));
    }

    @Test
    @Disabled("demo apikey has quotas")
    void testJSONStructuredOpenAI() throws Exception {
        // GIVEN
        RunContext runContext = runContextFactory.of(
            Map.of(
                "prompt", "Hello, my name is John. I was born on January 1, 2000.",
                "systemMessage", "You extract structured JSON data from text following the given schema.",
                "jsonFields", List.of("name", "date"),
                "schemaName", "Person",
                "modelName", "gpt-4o-mini",
                "apiKey", "demo",
                "baseUrl", "http://langchain4j.dev/demo/openai/v1"
            )
        );

        JSONStructuredExtraction task = JSONStructuredExtraction.builder()
            .prompt(Property.ofExpression("{{ prompt }}"))
            .systemMessage(Property.ofExpression("{{ systemMessage }}"))
            .schemaName(Property.ofExpression("{{ schemaName }}"))
            .jsonFields(Property.ofExpression("{{ jsonFields }}"))
            .provider(
                OpenAI.builder()
                    .type(OpenAI.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build()
            )
            .build();

        // WHEN
        JSONStructuredExtraction.Output runOutput = task.run(runContext);

        // THEN
        assertThat(runOutput.getExtractedJson(), notNullValue());
        JsonNode json = JacksonMapper.ofJson().readTree(runOutput.getExtractedJson());
        assertThat(json.has("name"), is(true));
        assertThat(json.has("date"), is(true));
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "OPENROUTER_API_KEY", matches = ".*")
    void withInputGuardrailPasses() throws Exception {
        RunContext runContext = runContextFactory.of(
            Map.of(
                "apiKey", OPENROUTER_API_KEY,
                "modelName", "openrouter/free",
                "baseUrl", "https://openrouter.ai/api/v1"
            )
        );

        JSONStructuredExtraction task = JSONStructuredExtraction.builder()
            .prompt(Property.ofValue("Hello, my name is John. I was born on January 1, 2000."))
            .schemaName(Property.ofValue("Person"))
            .jsonFields(Property.ofValue(List.of("name", "date")))
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
                                .build()
                        )
                    )
                    .build()
            )
            .build();

        JSONStructuredExtraction.Output output = task.run(runContext);

        assertThat(output.isGuardrailViolated(), is(false));
        assertThat(output.getExtractedJson(), notNullValue());
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
        JSONStructuredExtraction task = JSONStructuredExtraction.builder()
            .prompt(Property.ofValue("Hello World"))
            .schemaName(Property.ofValue("Person"))
            .jsonFields(Property.ofValue(List.of("name")))
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

        JSONStructuredExtraction.Output output = task.run(runContext);

        assertThat(output.isGuardrailViolated(), is(true));
        assertThat(output.getGuardrailViolationMessage(), containsString("Prompt too long"));
        assertThat(output.getExtractedJson(), nullValue());
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

        JSONStructuredExtraction task = JSONStructuredExtraction.builder()
            .prompt(Property.ofValue("Hello, my name is John. I was born on January 1, 2000."))
            .schemaName(Property.ofValue("Person"))
            .jsonFields(Property.ofValue(List.of("name", "date")))
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

        JSONStructuredExtraction.Output output = task.run(runContext);

        assertThat(output.isGuardrailViolated(), is(false));
        assertThat(output.getExtractedJson(), notNullValue());
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
        JSONStructuredExtraction task = JSONStructuredExtraction.builder()
            .prompt(Property.ofValue("Hello, my name is John. I was born on January 1, 2000."))
            .schemaName(Property.ofValue("Person"))
            .jsonFields(Property.ofValue(List.of("name", "date")))
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

        JSONStructuredExtraction.Output output = task.run(runContext);

        assertThat(output.isGuardrailViolated(), is(true));
        assertThat(output.getGuardrailViolationMessage(), containsString("Response contains confidential information"));
        assertThat(output.getExtractedJson(), nullValue());
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
        JSONStructuredExtraction task = JSONStructuredExtraction.builder()
            .prompt(Property.ofValue("Hello World"))
            .schemaName(Property.ofValue("Person"))
            .jsonFields(Property.ofValue(List.of("name")))
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

        JSONStructuredExtraction.Output output = task.run(runContext);

        assertThat(output.isGuardrailViolated(), is(true));
        assertThat(output.getGuardrailViolationMessage(), containsString("Prompt exceeds strict limit"));
        assertThat(output.getGuardrailViolationMessage(), not(containsString("Should never be reached")));
        assertThat(output.getExtractedJson(), nullValue());
    }
}
