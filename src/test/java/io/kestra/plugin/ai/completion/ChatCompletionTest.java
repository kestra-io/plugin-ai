package io.kestra.plugin.ai.completion;

import dev.langchain4j.model.anthropic.AnthropicChatModelName;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.workersai.WorkersAiChatModelName;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.ContainerTest;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;
import io.kestra.plugin.ai.provider.*;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.time.ZoneId;
import java.util.List;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

@KestraTest
class ChatCompletionTest extends ContainerTest {
    private final String GEMINI_API_KEY = System.getenv("GEMINI_API_KEY");
    private final String ANTHROPIC_API_KEY = System.getenv("ANTHROPIC_API_KEY");
    private final String MISTRAL_API_KEY = System.getenv("MISTRAL_API_KEY");
    private final String DEEPSEEK_API_KEY = System.getenv("DEEPSEEK_API_KEY");
    private final String OPENROUTER_API_KEY = System.getenv("OPENROUTER_API_KEY");
    private final String AMAZON_ACCESS_KEY_ID = System.getenv("AWS_ACCESS_KEY_ID");
    private final String AMAZON_SECRET_ACCESS_KEY = System.getenv("AWS_SECRET_ACCESS_KEY");
    private final String AZURE_OPENAI_API_KEY = System.getenv("AZURE_OPENAI_API_KEY");
    private final String WORKERS_AI_ACCOUNT_ID = System.getenv("WORKERS_AI_ACCOUNT_ID");
    private final String WORKERS_AI_API_KEY = System.getenv("WORKERS_AI_API_KEY");
    private final String DASHSCOPE_API_KEY = System.getenv("DASHSCOPE_API_KEY");
    private final String DASHSCOPE_CN_URL = "https://dashscope.aliyuncs.com/api/v1";
    private final String DASHSCOPE_INTL_URL = "https://dashscope-intl.aliyuncs.com/api/v1";
    private final String DASHSCOPE_BASE_URL =
          ZoneId.systemDefault().equals(ZoneId.of("Asia/Shanghai"))
              ? DASHSCOPE_CN_URL
              : DASHSCOPE_INTL_URL;

    @Inject
    private RunContextFactory runContextFactory;

    /**
     * Test Chat Completion using OpenAI.
     */
    @Test
    @Disabled("demo apikey has quotas")
    void testChatCompletionOpenAI() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
    }

    /**
     * Test Chat Completion using Gemini.
     */
    @Test
    @EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
    void testChatCompletionGemini() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", GEMINI_API_KEY,
            "modelName", "gemini-2.0-flash",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .messages(Property.ofExpression("{{ messages }}"))
            .provider(GoogleGemini.builder()
                .type(GoogleGemini.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
        assertThat(output.getSources(), notNullValue());
        assertTrue(output.getSources().isEmpty());
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
    void testChatCompletionGemini_givenMaxTokenInput_shouldRespectMaxOutputTokens() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", GEMINI_API_KEY,
            "modelName", "gemini-2.0-flash",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789))
                .maxToken(Property.ofValue(10)).build())
            .messages(Property.ofExpression("{{ messages }}"))
            .provider(GoogleGemini.builder()
                .type(GoogleGemini.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
        assertThat(output.getSources(), notNullValue());
        assertTrue(output.getSources().isEmpty());
        assertThat(output.getTokenUsage().getOutputTokenCount(), equalTo(10));
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
    void testChatCompletionGemini_givenThinkingConfiguration() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", GEMINI_API_KEY,
            "modelName", "gemini-2.5-flash",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));
        ChatCompletion task = ChatCompletion.builder()
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789))
                .thinkingEnabled(Property.ofValue(true)).thinkingBudgetTokens(Property.ofValue(1024))
                .returnThinking(Property.ofValue(true)).build())
            .messages(Property.ofExpression("{{ messages }}"))
            .provider(GoogleGemini.builder()
                .type(GoogleGemini.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);
        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
        assertThat(output.getThinking(), notNullValue());
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
    void testGeminiChatCompletion_givenThinkingEnabled_butReturnThinkingDisabled() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", GEMINI_API_KEY,
            "modelName", "gemini-2.5-flash",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));
        ChatCompletion task = ChatCompletion.builder()
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789))
                .thinkingEnabled(Property.ofValue(true)).thinkingBudgetTokens(Property.ofValue(1024)).build())
            .messages(Property.ofExpression("{{ messages }}"))
            .provider(GoogleGemini.builder()
                .type(GoogleGemini.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);
        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
        assertThat(output.getThinking(), isEmptyOrNullString());
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
    void testChatCompletionGemini_givenInvalidModel_whenThinkingNotAllowed_thenThrowException() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", GEMINI_API_KEY,
            "modelName", "gemini-2.0-flash",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));
        ChatCompletion task = ChatCompletion.builder()
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789))
                .thinkingEnabled(Property.ofValue(true)).thinkingBudgetTokens(Property.ofValue(1024)).build())
            .messages(Property.ofExpression("{{ messages }}"))
            .provider(GoogleGemini.builder()
                .type(GoogleGemini.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .build()
            )
            .build();

        // Assert RuntimeException and error message
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            ChatCompletion.Output output = task.run(runContext);
        }, "status code: 400");

        // Verify error message contains 404 details
        assertThat(exception.getMessage(), containsString("Unable to submit request because thinking is not supported by this model."));
    }


    /**
     * Test Chat Completion using Ollama.
     */
    @Test
    void testChatCompletionOllama() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "tinydolphin",
            "ollamaEndpoint", ollamaEndpoint,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(Ollama.builder()
                .type(Ollama.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .endpoint(Property.ofExpression("{{ ollamaEndpoint }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
    }

    @Test
    void testChatCompletionOllama_givenInvalidModel_whenThinkingNotAllowed_throwsException() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "mistral",
            "ollamaEndpoint", "tinydolphin",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789))
                .thinkingEnabled(Property.ofValue(true)).build())
            .provider(Ollama.builder()
                .type(Ollama.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .endpoint(Property.ofExpression("{{ ollamaEndpoint }}"))
                .build()
            )
            .build();

        // Assert RuntimeException and error message
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            ChatCompletion.Output output = task.run(runContext);
        }, "status code: 400");

    }

    @Disabled
    @Test
    void testChatCompletionOllama_givenThinkingConfigurationEnabled() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "llama3",
            "ollamaEndpoint", ollamaEndpoint,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789))
                .thinkingEnabled(Property.ofValue(true)).build())
            .provider(Ollama.builder()
                .type(Ollama.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .endpoint(Property.ofExpression("{{ ollamaEndpoint }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
    }


    @Test
    void testChatCompletionStructuredOutput() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "tinydolphin",
            "ollamaEndpoint", ollamaEndpoint,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John. I was born on January 1, 2000.").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder()
                .temperature(Property.ofValue(0.1))
                .seed(Property.ofValue(123456789))
                .responseFormat(ChatConfiguration.ResponseFormat.builder()
                    .type(Property.ofValue(ResponseFormatType.JSON))
                    .jsonSchema(Property.ofValue(
                        Map.of(
                            "type", "object",
                            "properties", Map.of(
                                "name", Map.of("type", "string"),
                                "birth", Map.of("type", "string")
                            )
                        )
                    ))
                    .build()
                )
                .build()
            )
            .provider(Ollama.builder()
                .type(Ollama.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .endpoint(Property.ofExpression("{{ ollamaEndpoint }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        System.out.println(output.getJsonOutput());
        assertThat(output.getJsonOutput(),  aMapWithSize(2));

    }

    @Test
    void testChatCompletionNoTemplate() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "tinydolphin",
            "ollamaEndpoint", ollamaEndpoint,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is {{John}}").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(Ollama.builder()
                .type(Ollama.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .endpoint(Property.ofExpression("{{ ollamaEndpoint }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
    }

    @Test
    void shouldThrowWhenMoreThanOneSystemMessage() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "tinydolphin",
            "ollamaEndpoint", ollamaEndpoint,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.SYSTEM).content("You are a bot").build(),
                ChatMessage.builder().type(ChatMessageType.SYSTEM).content("You are an alien").build(),
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(Ollama.builder()
                .type(Ollama.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .endpoint(Property.ofExpression("{{ ollamaEndpoint }}"))
                .build()
            )
            .build();

        assertThrows(IllegalArgumentException.class, () -> task.run(runContext));
    }

    @Test
    void shouldThrowWhenLastMessageIsNotUserMessage() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "tinydolphin",
            "ollamaEndpoint", ollamaEndpoint,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.SYSTEM).content("You are a bot").build(),
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build(),
                ChatMessage.builder().type(ChatMessageType.AI).content("You are an alien").build()
                )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(Ollama.builder()
                .type(Ollama.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .endpoint(Property.ofExpression("{{ ollamaEndpoint }}"))
                .build()
            )
            .build();

        assertThrows(IllegalArgumentException.class, () -> task.run(runContext));
    }


    @EnabledIfEnvironmentVariable(named = "ANTHROPIC_API_KEY", matches = ".*")
    @Test
    void testChatCompletionAnthropicAI() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", AnthropicChatModelName.CLAUDE_3_HAIKU_20240307,
            "apiKey", ANTHROPIC_API_KEY,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(Anthropic.builder()
                .type(Anthropic.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
    }
    @EnabledIfEnvironmentVariable(named = "ANTHROPIC_API_KEY", matches = ".*")
    @Test
    void testChatCompletionAnthropicAI_givenMaxTokenInput_shouldRespectMaxOutputTokens() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", AnthropicChatModelName.CLAUDE_3_HAIKU_20240307,
            "apiKey", ANTHROPIC_API_KEY,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1))
                .maxToken(Property.ofValue(10)).build())
            .provider(Anthropic.builder()
                .type(Anthropic.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
        assertThat(output.getTokenUsage().getOutputTokenCount(), equalTo(10));
    }


    @Test
    void testChatCompletionAnthropicAI_givenInvalidApiKey_shouldThrow4xxUnAuthorizedException() {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", AnthropicChatModelName.CLAUDE_3_HAIKU_20240307,
            "apiKey", "DUMMY_ANTHROPIC_API_KEY",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).build())
            .provider(Anthropic.builder()
                .type(Anthropic.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .build()
            )
            .build();

        // Assert RuntimeException and error message
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            ChatCompletion.Output output = task.run(runContext);
        }, "status code: 401");

        // Verify error message contains 404 details
        assertThat(exception.getMessage(), containsString("authentication_error"));
    }
    @EnabledIfEnvironmentVariable(named = "ANTHROPIC_API_KEY", matches = ".*")
    @Test
    void testChatCompletionAnthropicAI_givenThinkingConfiguration() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", ANTHROPIC_API_KEY,
            "modelName", AnthropicChatModelName.CLAUDE_SONNET_4_20250514,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));
        ChatCompletion task = ChatCompletion.builder()
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(1.0))
                .thinkingEnabled(Property.ofValue(true)).thinkingBudgetTokens(Property.ofValue(1024))
                .returnThinking(Property.ofValue(true)).build())
            .messages(Property.ofExpression("{{ messages }}"))
            .provider(Anthropic.builder()
                .type(Anthropic.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .maxTokens(Property.ofValue(2024))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
        assertThat(output.getThinking(), notNullValue());
    }

    @EnabledIfEnvironmentVariable(named = "ANTHROPIC_API_KEY", matches = ".*")
    @Test
    void testChatCompletionAnthropicAI_givenThinkingEnabled_butReturnThinkingDisabled() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", ANTHROPIC_API_KEY,
            "modelName", AnthropicChatModelName.CLAUDE_SONNET_4_20250514,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));
        ChatCompletion task = ChatCompletion.builder()
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(1.0))
                .thinkingEnabled(Property.ofValue(true)).thinkingBudgetTokens(Property.ofValue(1024)).build())
            .messages(Property.ofExpression("{{ messages }}"))
            .provider(Anthropic.builder()
                .type(Anthropic.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .maxTokens(Property.ofValue(2024))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
        assertThat(output.getThinking(), isEmptyOrNullString());
    }
    @EnabledIfEnvironmentVariable(named = "ANTHROPIC_API_KEY", matches = ".*")
    @Test
    void testChatCompletionAnthropicAI_givenThinkingEnabled_whenMaxTokensLessThanMaxToken_thenThrowException() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", ANTHROPIC_API_KEY,
            "modelName", AnthropicChatModelName.CLAUDE_SONNET_4_20250514,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));
        ChatCompletion task = ChatCompletion.builder()
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(1.0))
                .thinkingEnabled(Property.ofValue(true)).thinkingBudgetTokens(Property.ofValue(1025)).build())
            .messages(Property.ofExpression("{{ messages }}"))
            .provider(Anthropic.builder()
                .type(Anthropic.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .build()
            )
            .build();

        // Assert RuntimeException and error message
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            ChatCompletion.Output output = task.run(runContext);
        }, "status code: 400");

        // Verify error message contains 404 details
        assertThat(exception.getMessage(), containsString("`max_tokens` must be greater than `thinking.budget_tokens` for thinking-enabled Anthropic models."));
    }
    @EnabledIfEnvironmentVariable(named = "MISTRAL_API_KEY", matches = ".*")
    @Test
    void testChatCompletionMistralAI() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "mistral:7b",
            "apiKey", MISTRAL_API_KEY,
            "baseUrl", "https://api.mistral.ai/v1",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(MistralAI.builder()
                .type(MistralAI.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
    }

    @Test
    void testChatCompletionMistralAI_givenInvalidApiKey_shouldThrow4xxUnAuthorizedException() {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "mistral:7b",
            "apiKey", "DUMMY_MISTRAL_API_KEY",
            "baseUrl", "https://api.mistral.ai/v1",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(MistralAI.builder()
                .type(MistralAI.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .build();

        // Assert RuntimeException and error message
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            ChatCompletion.Output output = task.run(runContext);
        }, "status code: 401");

        // Verify error message contains 404 details
        assertThat(exception.getMessage(), containsString("Unauthorized"));
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "MISTRAL_API_KEY", matches = ".*")
    void testChatCompletionMistralAI_givenInvalidBaseUrlMistralAI_shouldThrow4xx() {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "mistral:7b",
            "apiKey", MISTRAL_API_KEY,
            "baseUrl", ollamaEndpoint,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(MistralAI.builder()
                .type(MistralAI.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .build();

        // Assert RuntimeException and error message
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            ChatCompletion.Output output = task.run(runContext);
        }, "status code: 404; body: 404 page not found");

        // Verify error message contains 404 details
        assertThat(exception.getMessage(), containsString("404"));
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "DEEPSEEK_API_KEY", matches = ".*")
    void testChatCompletionDeepseek() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", DEEPSEEK_API_KEY,
            "modelName", "deepseek-chat",
            "baseUrl", "https://api.deepseek.com/v1",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(DeepSeek.builder()
                .type(DeepSeek.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "DEEPSEEK_API_KEY", matches = ".*")
    void testChatCompletionDeepseek_givenThinkingConfiguration() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "sk-d7204degb5f46f3cab6730c1108e2defa",
            "modelName", "deepseek-chat",
            "baseUrl", "https://api.deepseek.com/v1",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789))
                .thinkingEnabled(Property.ofValue(true)).build())
            .provider(DeepSeek.builder()
                .type(DeepSeek.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
    }

    @Test
    void testChatCompletionDeepseek_givenInvalidApiKey_shouldThrow4xxUnAuthorizedException() {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "DUMMY_DEEPSEEK_API_KEY",
            "modelName", "deepseek-chat",
            "baseUrl", "https://api.deepseek.com/v1",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(DeepSeek.builder()
                .type(DeepSeek.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .build();

        // Assert RuntimeException and error message
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            ChatCompletion.Output output = task.run(runContext);
        }, "status code: 401");

        // Verify error message contains 404 details
        assertThat(exception.getMessage(), containsString("Authentication Fails, Your api key: ****_KEY is invalid"));
    }


    @EnabledIfEnvironmentVariable(named = "AWS_ACCESS_KEY_ID", matches = ".*")
    @EnabledIfEnvironmentVariable(named = "AWS_SECRET_ACCESS_KEY", matches = ".*")
    @Test
    void testChatCompletionAmazonBedrockAI() throws Exception {
        String modelName = "anthropic.claude-3-sonnet-20240229-v1:0";
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", modelName,
            "accessKeyId", AMAZON_ACCESS_KEY_ID,
            "secretAccessKey", AMAZON_SECRET_ACCESS_KEY,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(AmazonBedrock.builder()
                .type(AmazonBedrock.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .accessKeyId(Property.ofExpression("{{ accessKeyId }}"))
                .secretAccessKey(Property.ofExpression("{{ secretAccessKey }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
    }

    @Test
    void testChatCompletionAmazonBedrockAI_givenInvalidApiKey_shouldThrow4xxUnAuthorizedException() {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "anthropic.claude-3-sonnet-20240229-v1:0",
            "accessKeyId", "DUMMY_ACCESS_KEY_ID",
            "secretAccessKey", "DUMMY_SECRET_ACCESS_KEY",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).build())
            .provider(AmazonBedrock.builder()
                .type(AmazonBedrock.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .accessKeyId(Property.ofExpression("{{ accessKeyId }}"))
                .secretAccessKey(Property.ofExpression("{{ secretAccessKey }}"))
                .build()
            )
            .build();

        // Assert RuntimeException and error message
        RuntimeException exception = assertThrows(RuntimeException.class, () -> task.run(runContext), "status code: 401");

        // Verify error message
        assertThat(exception.getMessage(), containsString("Unable to load region from any of the providers in the chain"));
    }


    @EnabledIfEnvironmentVariable(named = "AZURE_OPENAI_API_KEY", matches = ".*")
    @Test
    void testChatCompletionAzureOpenAI() throws Exception {
        String modelName = "anthropic.claude-3-sonnet-20240229-v1:0";
        String azureEndpoint = "https://kestra.openai.azure.com/";
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", modelName,
            "apiKey", AZURE_OPENAI_API_KEY,
            "endpoint", azureEndpoint,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(AzureOpenAI.builder()
                .type(AzureOpenAI.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .endpoint(Property.ofExpression("{{ endpoint }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
    }

    @Test
    void testChatCompletionAzureOpenAI_givenInvalidApiKey_shouldThrow4xxUnAuthorizedException() {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "anthropic.claude-3-sonnet-20240229-v1:0",
            "apiKey", "DUMMY_API_KEY",
            "endpoint", "https://kestra.openai.azure.com/",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).build())
            .provider(AzureOpenAI.builder()
                .type(AzureOpenAI.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .endpoint(Property.ofExpression("{{ endpoint }}"))
                .build()
            )
            .build();

        // Assert RuntimeException and error message
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            ChatCompletion.Output output = task.run(runContext);
        }, "status code: 401");

        // Verify error message
        assertThat(exception.getMessage(), containsString("UnknownHostException"));
    }


    @Test
    @EnabledIfEnvironmentVariable(named = "OPENROUTER_API_KEY", matches = ".*")
    void testChatCompletionOpenRouter() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", OPENROUTER_API_KEY,
            "modelName", "mistralai/mistral-7b-instruct:free",
                 "baseUrl", "https://openrouter.ai/api/v1",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(OpenRouter.builder()
                .type(OpenRouter.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
    }

    @Test
    void testChatCompletionOpenRouter_givenInvalidApiKey_shouldThrow4xxUnAuthorizedException() {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "OPENROUTER_API_KEY",
            "modelName", "deepseek/deepseek-r1:free",
            "baseUrl", "https://openrouter.ai/api/v1",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(OpenRouter.builder()
                .type(OpenRouter.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .build();

        // Assert RuntimeException and error message
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            ChatCompletion.Output output = task.run(runContext);
        }, "status code: 401");

        // Verify error message contains authentication error details
        assertThat(exception.getMessage(), containsString("401"));
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "OPENROUTER_API_KEY", matches = ".*")
    void testChatCompletionOpenRouter_withDefaultBaseUrl() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", OPENROUTER_API_KEY,
            "modelName", "mistralai/mistral-7b-instruct:free",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .provider(OpenRouter.builder()
                .type(OpenRouter.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                // Note: baseUrl not specified, should use default
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
    }

    @Test
   @EnabledIfEnvironmentVariable(named = "OPENROUTER_API_KEY", matches = ".*")
    void testChatCompletionOpenRouter_givenMaxTokenInput_shouldRespectMaxOutputTokens() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", OPENROUTER_API_KEY,
            "modelName", "mistralai/mistral-7b-instruct:free",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789))
                .maxToken(Property.ofValue(10)).build())
            .provider(OpenRouter.builder()
                .type(OpenRouter.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                // Note: baseUrl not specified, should use default
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
        assertThat(output.getRequestDuration(), notNullValue());
        assertThat(output.getTokenUsage().getOutputTokenCount(), equalTo(10));
    }
  @Test
  @EnabledIfEnvironmentVariable(named = "DASHSCOPE_API_KEY", matches = ".*")
  void testChatCompletionDashScope() throws Exception {
    RunContext runContext =
        runContextFactory.of(
            Map.of(
                "apiKey", DASHSCOPE_API_KEY,
                "modelName", "qwen-plus",
                "baseUrl", DASHSCOPE_BASE_URL,
                "messages", List.of(
                        ChatMessage.builder()
                            .type(ChatMessageType.USER)
                            .content("Hello, my name is John")
                            .build())));

    ChatCompletion task =
        ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(
                ChatConfiguration.builder()
                    .temperature(Property.ofValue(0.1))
                    .seed(Property.ofValue(123456789))
                    .build())
            .provider(
                DashScope.builder()
                    .type(DashScope.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build())
            .build();

    ChatCompletion.Output output = task.run(runContext);

    assertThat(output.getTextOutput(), notNullValue());
    assertThat(output.getTextOutput(), containsString("John"));
    assertThat(output.getRequestDuration(), notNullValue());
  }

  @Test
  @EnabledIfEnvironmentVariable(named = "DASHSCOPE_API_KEY", matches = ".*")
  void testChatCompletionDashScope_givenThinkingConfiguration() throws Exception {
    RunContext runContext =
        runContextFactory.of(
            Map.of(
                "apiKey", DASHSCOPE_API_KEY,
                "modelName", "qwen-plus",
                "baseUrl", DASHSCOPE_BASE_URL,
                "messages", List.of(
                        ChatMessage.builder()
                            .type(ChatMessageType.USER)
                            .content("Hello, my name is John")
                            .build())));

    ChatCompletion task =
        ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(
                ChatConfiguration.builder()
                    .temperature(Property.ofValue(0.1))
                    .seed(Property.ofValue(123456789))
                    .thinkingEnabled(Property.ofValue(true))
                    .build())
            .provider(
                DashScope.builder()
                    .type(DashScope.class.getName())
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build())
            .build();

    ChatCompletion.Output output = task.run(runContext);

    assertThat(output.getTextOutput(), notNullValue());
    assertThat(output.getTextOutput(), containsString("John"));
    assertThat(output.getRequestDuration(), notNullValue());
  }

  @Test
  void testChatCompletionDashScope_givenInvalidApiKey_shouldThrow4xxUnAuthorizedException() {
    RunContext runContext =
        runContextFactory.of(
            Map.of(
                "apiKey", "DUMMY_DASHSCOPE_API_KEY",
                "modelName", "qwen-plus",
                "baseUrl", DASHSCOPE_BASE_URL,
                "messages", List.of(
                        ChatMessage.builder()
                            .type(ChatMessageType.USER)
                            .content("Hello, my name is John")
                            .build())));

    ChatCompletion task =
        ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(
                ChatConfiguration.builder()
                    .temperature(Property.ofValue(0.1))
                    .seed(Property.ofValue(123456789))
                    .build())
            .provider(
                DashScope.builder()
                    .type(DashScope.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                    .build())
            .build();

    // Assert RuntimeException and error message
    RuntimeException exception =
        assertThrows(
            RuntimeException.class,
            () -> {
              ChatCompletion.Output output = task.run(runContext);
            },
            "status code: 401");

    // Verify error message contains 404 details
    assertThat(exception.getMessage(), containsString("Invalid API-key provided."));
  }

    @EnabledIfEnvironmentVariable(named = "WORKERS_AI_API_KEY", matches = ".*")
    @EnabledIfEnvironmentVariable(named = "WORKERS_AI_ACCOUNT_ID", matches = ".*")
    @Test
    void testChatCompletionWorkersAI_givenMaxTokenInput_shouldRespectMaxOutputTokens() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", WorkersAiChatModelName.LLAMA2_7B_FULL,
            "apiKey", WORKERS_AI_API_KEY,
            "accountId", WORKERS_AI_ACCOUNT_ID,
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            .provider(WorkersAI.builder()
                .type(WorkersAI.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .accountId(Property.ofExpression("{{ accountId }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
    }


    @Test
    void testChatCompletionWorkersAI_givenInvalidApiKey_shouldThrow4xxUnAuthorizedException() {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", WorkersAiChatModelName.LLAMA2_7B_FULL,
            "apiKey", "WORKERS_AI_API_KEY",
            "accountId", "WORKERS_AI_ACCOUNT_ID",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            .provider(WorkersAI.builder()
                .type(WorkersAI.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .accountId(Property.ofExpression("{{ accountId }}"))
                .build()
            )
            .build();

        // Assert RuntimeException and error message
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            ChatCompletion.Output output = task.run(runContext);
        }, "status code: 401");

        // Verify error message contains 404 details
        assertThat(exception.getMessage(), containsString("WORKERS_AI_ACCOUNT_ID"));
    }

    @Disabled
    @Test
    void testChatCompletionLocalAI_givenMaxTokenInput_shouldRespectMaxOutputTokens() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "gemma-3-1b-it",
            "baseUrl", "http://localhost:8080/v1",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()
            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1))
                .maxToken(Property.ofValue(10)).build())
            .provider(LocalAI.builder()
                .type(LocalAI.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .build();

        ChatCompletion.Output output = task.run(runContext);

        assertThat(output.getTextOutput(), notNullValue());
        assertThat(output.getTextOutput(), containsString("John"));
    }


    @Test
    void testChatCompletionLocalAI_givenInvalidApiKey_shouldThrow4xxUnAuthorizedException() {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "gemma-3-1b-it",
            "baseUrl", "http://localhost/v1",
            "messages", List.of(
                ChatMessage.builder().type(ChatMessageType.USER).content("Hello, my name is John").build()

            )
        ));

        ChatCompletion task = ChatCompletion.builder()
            .messages(Property.ofExpression("{{ messages }}"))
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1))
                .maxToken(Property.ofValue(10)).build())
              .provider(LocalAI.builder()
                .type(LocalAI.class.getName())
                .modelName(Property.ofExpression("{{ modelName }}"))
                  .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .build();

        // Assert RuntimeException and error message
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            ChatCompletion.Output output = task.run(runContext);
        }, "status code: 401");

        // Verify error message contains 404 details
        assertThat(exception.getMessage(), containsString("java.net.ConnectException"));
    }
}
