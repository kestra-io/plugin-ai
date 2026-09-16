package io.kestra.plugin.ai;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.Task;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.kestra.plugin.ai.domain.ToolProvider;
import io.kestra.plugin.ai.provider.GoogleGemini;
import io.kestra.plugin.ai.provider.OpenAI;
import io.kestra.plugin.ai.tool.KestraFlow;

import jakarta.inject.Inject;
import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validator;

import dev.langchain4j.model.chat.request.ResponseFormatType;

import static org.assertj.core.api.Assertions.assertThat;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class GeminiConfigurationValidationTest {
    private static final String VALIDATION_MESSAGE = "GoogleGemini provider does not support combining 'tools' with 'responseFormat'";

    @Inject
    private Validator validator;

    @Test
    void rejectsToolsCombinedWithResponseFormatForGemini() {
        tasks(gemini(), responseFormat(), List.of(tool())).forEach(task ->
            assertThat(conflictViolations(task))
                .singleElement()
                .extracting(ConstraintViolation::getMessage)
                .isEqualTo(VALIDATION_MESSAGE)
        );
    }

    @Test
    void acceptsSupportedCombinations() {
        tasks(gemini(), ChatConfiguration.empty(), List.of(tool()))
            .forEach(task -> assertThat(conflictViolations(task)).isEmpty());
        tasks(gemini(), responseFormat(), null)
            .forEach(task -> assertThat(conflictViolations(task)).isEmpty());
        tasks(openAI(), responseFormat(), List.of(tool()))
            .forEach(task -> assertThat(conflictViolations(task)).isEmpty());
    }

    private List<Task> tasks(ModelProvider provider, ChatConfiguration configuration, List<ToolProvider> tools) {
        return List.of(
            io.kestra.plugin.ai.agent.AIAgent.builder()
                .prompt(Property.ofValue("prompt"))
                .provider(provider)
                .configuration(configuration)
                .tools(tools)
                .build(),
            io.kestra.plugin.ai.completion.ChatCompletion.builder()
                .messages(Property.ofExpression("{{ messages }}"))
                .provider(provider)
                .configuration(configuration)
                .tools(tools)
                .build(),
            io.kestra.plugin.ai.rag.ChatCompletion.builder()
                .prompt(Property.ofValue("prompt"))
                .chatProvider(provider)
                .chatConfiguration(configuration)
                .tools(tools)
                .build()
        );
    }

    private List<ConstraintViolation<Task>> conflictViolations(Task task) {
        return validator.validate(task).stream()
            .filter(violation -> VALIDATION_MESSAGE.equals(violation.getMessage()))
            .toList();
    }

    private GoogleGemini gemini() {
        return GoogleGemini.builder()
            .modelName(Property.ofValue("gemini-3.5-flash"))
            .apiKey(Property.ofValue("api-key"))
            .build();
    }

    private OpenAI openAI() {
        return OpenAI.builder()
            .modelName(Property.ofValue("gpt-4o-mini"))
            .apiKey(Property.ofValue("api-key"))
            .build();
    }

    private ChatConfiguration responseFormat() {
        return ChatConfiguration.builder()
            .responseFormat(
                ChatConfiguration.ResponseFormat.builder()
                    .type(Property.ofValue(ResponseFormatType.JSON))
                    .build()
            )
            .build();
    }

    private ToolProvider tool() {
        return KestraFlow.builder()
            .namespace(Property.ofValue("company.team"))
            .flowId(Property.ofValue("hello-world"))
            .description(Property.ofValue("Say hello"))
            .build();
    }
}
