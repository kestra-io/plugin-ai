package io.kestra.plugin.ai.completion;

import dev.langchain4j.model.chat.request.ResponseFormatType;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;
import io.kestra.plugin.ai.provider.OpenAI;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.util.List;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.empty;

@KestraTest
class ChatCompletionStrictJsonModeIntegrationTest {
    private final String OPENAI_API_KEY = System.getenv("OPENAI_API_KEY");
    private final String OPENAI_MODEL = System.getenv().getOrDefault("OPENAI_MODEL", "gpt-4o-mini");

    @Inject
    private RunContextFactory runContextFactory;

    @Test
    @EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".*")
    void testChatCompletionOpenAI_givenStrictJsonModeAndArraySchema_shouldReturnStructuredPeopleList() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", OPENAI_API_KEY,
            "modelName", OPENAI_MODEL,
            "messages", List.of(
                ChatMessage.builder()
                    .type(ChatMessageType.USER)
                    .content("Generate exactly 3 fictional people with silly favorite hobbies.")
                    .build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            .configuration(ChatConfiguration.builder()
                .responseFormat(ChatConfiguration.ResponseFormat.builder()
                    .type(Property.ofValue(ResponseFormatType.JSON))
                    .jsonSchema(Property.ofValue(Map.of(
                        "type", "object",
                        "properties", Map.of(
                            "people", Map.of(
                                "type", "array",
                                "items", Map.of(
                                    "type", "object",
                                    "properties", Map.of(
                                        "name", Map.of("type", "string"),
                                        "hobby", Map.of("type", "string")
                                    ),
                                    "required", List.of("name", "hobby")
                                )
                            )
                        ),
                        "required", List.of("people")
                    )))
                    .build())
                .build())
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .enableStrictJson(Property.ofValue(true))
                .build())
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getJsonOutput(), notNullValue());
        assertThat(output.getJsonOutput().get("people"), instanceOf(List.class));

        List<?> people = (List<?>) output.getJsonOutput().get("people");
        assertThat(people, not(empty()));

        for (Object person : people) {
            assertThat(person, instanceOf(Map.class));
            Map<?, ?> personMap = (Map<?, ?>) person;
            assertThat(personMap.get("name"), instanceOf(String.class));
            assertThat(personMap.get("hobby"), instanceOf(String.class));
        }
    }
}
