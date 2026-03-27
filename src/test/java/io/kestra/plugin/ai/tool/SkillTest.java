package io.kestra.plugin.ai.tool;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import com.github.tomakehurst.wiremock.WireMockServer;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ChatMessage;
import io.kestra.plugin.ai.domain.ChatMessageType;
import io.kestra.plugin.ai.provider.OpenAI;

import dev.langchain4j.model.output.FinishReason;
import jakarta.inject.Inject;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig;
import static org.assertj.core.api.Assertions.assertThat;

@KestraTest(startRunner = true)
class SkillTest {
    @Inject
    private RunContextFactory runContextFactory;

    @Test
    void skillWithInlineContent() throws Exception {
        var wireMock = new WireMockServer(wireMockConfig().dynamicPort());
        try {
            wireMock.start();

            // First call: LLM decides to call activate_skill tool
            wireMock.stubFor(post(urlEqualTo("/v1/chat/completions"))
                .inScenario("skill-activation")
                .whenScenarioStateIs("Started")
                .willReturn(aResponse()
                    .withStatus(200)
                    .withHeader("Content-Type", "application/json")
                    .withBody("""
                        {
                          "id": "chatcmpl-1",
                          "object": "chat.completion",
                          "model": "gpt-4o-mini",
                          "choices": [{
                            "index": 0,
                            "message": {
                              "role": "assistant",
                              "content": null,
                              "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                  "name": "activate_skill",
                                  "arguments": "{\\"skill_name\\": \\"translator\\"}"
                                }
                              }]
                            },
                            "finish_reason": "tool_calls"
                          }],
                          "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
                        }
                        """))
                .willSetStateTo("skill-activated"));

            // Second call: LLM produces final answer after receiving skill content
            wireMock.stubFor(post(urlEqualTo("/v1/chat/completions"))
                .inScenario("skill-activation")
                .whenScenarioStateIs("skill-activated")
                .willReturn(aResponse()
                    .withStatus(200)
                    .withHeader("Content-Type", "application/json")
                    .withBody("""
                        {
                          "id": "chatcmpl-2",
                          "object": "chat.completion",
                          "model": "gpt-4o-mini",
                          "choices": [{
                            "index": 0,
                            "message": {
                              "role": "assistant",
                              "content": "Bonjour"
                            },
                            "finish_reason": "stop"
                          }],
                          "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
                        }
                        """)));

            RunContext runContext = runContextFactory.of(Map.of());

            var chat = ChatCompletion.builder()
                .provider(
                    OpenAI.builder()
                        .type(OpenAI.class.getName())
                        .apiKey(Property.ofValue("test-key"))
                        .modelName(Property.ofValue("gpt-4o-mini"))
                        .baseUrl(Property.ofValue(wireMock.baseUrl() + "/v1"))
                        .build()
                )
                .tools(
                    List.of(
                        Skill.builder()
                            .skills(
                                List.of(
                                    Skill.SkillDefinition.builder()
                                        .name(Property.ofValue("translator"))
                                        .description(Property.ofValue("Translates text"))
                                        .content(Property.ofValue("Translate to the target language."))
                                        .build()
                                )
                            )
                            .build()
                    )
                )
                .messages(
                    Property.ofValue(
                        List.of(
                            ChatMessage.builder()
                                .type(ChatMessageType.SYSTEM)
                                .content("Always activate a skill before answering.")
                                .build(),
                            ChatMessage.builder()
                                .type(ChatMessageType.USER)
                                .content("Translate 'Hello' to French.")
                                .build()
                        )
                    )
                )
                .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).build())
                .build();

            var output = chat.run(runContext);
            assertThat(output.getToolExecutions()).isNotEmpty();
            assertThat(output.getToolExecutions()).extracting("requestName").contains("activate_skill");
            assertThat(output.getTextOutput()).isEqualTo("Bonjour");
            assertThat(output.getIntermediateResponses()).isNotEmpty();
            assertThat(output.getIntermediateResponses().getFirst().getFinishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
        } finally {
            wireMock.stop();
        }
    }
}
