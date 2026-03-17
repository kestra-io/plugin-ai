package io.kestra.plugin.ai.guardrail;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.GuardrailRule;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class ExpressionOutputGuardrailTest {

    @Inject
    private TestRunContextFactory runContextFactory;

    private GuardrailRule rule(String expression, String message) {
        return GuardrailRule.builder().expression(expression).message(message).build();
    }

    private ExpressionOutputGuardrail guardrail(RunContext ctx, GuardrailRule... rules) {
        return new ExpressionOutputGuardrail(List.of(rules), ctx);
    }

    private ChatResponse response(String text, FinishReason finishReason, TokenUsage tokenUsage) {
        return ChatResponse.builder()
            .aiMessage(AiMessage.from(text))
            .finishReason(finishReason)
            .tokenUsage(tokenUsage)
            .build();
    }

    @Test
    void checkOrNull_response_passes() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(ctx, rule("{{ response.length > 0 }}", "violation"));

        assertThat(guardrail.checkOrNull(response("Hello world", FinishReason.STOP, new TokenUsage(10, 20)))).isNull();
    }

    @Test
    void checkOrNull_response_violated() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(ctx, rule("{{ response.length < 1 }}", "violation"));

        assertThat(guardrail.checkOrNull(response("Hello world", FinishReason.STOP, new TokenUsage(10, 20)))).isEqualTo("violation");
    }

    @Test
    void checkOrNull_finishReason_passes() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(ctx, rule("{{ finishReason == 'STOP' }}", "violation"));

        assertThat(guardrail.checkOrNull(response("Hello", FinishReason.STOP, new TokenUsage(10, 20)))).isNull();
    }

    @Test
    void checkOrNull_finishReason_violated() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(ctx, rule("{{ finishReason == 'LENGTH' }}", "violation"));

        assertThat(guardrail.checkOrNull(response("Hello", FinishReason.STOP, new TokenUsage(10, 20)))).isEqualTo("violation");
    }

    @Test
    void checkOrNull_nullFinishReason_defaultsToEmptyString() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(ctx, rule("{{ finishReason == '' }}", "violation"));

        assertThat(guardrail.checkOrNull(response("Hello", null, new TokenUsage(10, 20)))).isNull();
    }

    @Test
    void checkOrNull_inputTokenCount_passes() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(ctx, rule("{{ inputTokenCount > 0 }}", "violation"));

        assertThat(guardrail.checkOrNull(response("Hello", FinishReason.STOP, new TokenUsage(10, 20)))).isNull();
    }

    @Test
    void checkOrNull_inputTokenCount_violated() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(ctx, rule("{{ inputTokenCount > 999 }}", "violation"));

        assertThat(guardrail.checkOrNull(response("Hello", FinishReason.STOP, new TokenUsage(10, 20)))).isEqualTo("violation");
    }

    @Test
    void checkOrNull_outputTokenCount_passes() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(ctx, rule("{{ outputTokenCount > 0 }}", "violation"));

        assertThat(guardrail.checkOrNull(response("Hello", FinishReason.STOP, new TokenUsage(10, 20)))).isNull();
    }

    @Test
    void checkOrNull_outputTokenCount_violated() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(ctx, rule("{{ outputTokenCount > 999 }}", "violation"));

        assertThat(guardrail.checkOrNull(response("Hello", FinishReason.STOP, new TokenUsage(10, 20)))).isEqualTo("violation");
    }

    @Test
    void checkOrNull_nullTokenUsage_defaultsToZero() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(ctx, rule("{{ inputTokenCount == 0 and outputTokenCount == 0 }}", "violation"));

        assertThat(guardrail.checkOrNull(response("Hello", FinishReason.STOP, null))).isNull();
    }

    @Test
    void checkOrNull_nullResponseText_defaultsToEmpty() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(ctx, rule("{{ response == '' }}", "violation"));

        // AiMessage.from(null) or an empty AiMessage — use empty string
        ChatResponse chatResponse = ChatResponse.builder()
            .aiMessage(AiMessage.from(""))
            .finishReason(FinishReason.STOP)
            .tokenUsage(new TokenUsage(10, 20))
            .build();

        assertThat(guardrail.checkOrNull(chatResponse)).isNull();
    }

    @Test
    void checkOrNull_multipleRules_firstViolatingWins() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(
            ctx,
            rule("{{ response.length > 0 }}", "first"), // passes
            rule("{{ response.length < 1 }}", "second"), // fails
            rule("{{ response.length < 1 }}", "third") // never reached
        );

        assertThat(guardrail.checkOrNull(response("Hello", FinishReason.STOP, new TokenUsage(10, 20)))).isEqualTo("second");
    }

    @Test
    void checkOrNull_emptyRules_passes() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = new ExpressionOutputGuardrail(List.of(), ctx);

        assertThat(guardrail.checkOrNull(response("Hello", FinishReason.STOP, new TokenUsage(10, 20)))).isNull();
    }

    @Test
    void checkOrNull_allFourVariables_passes() throws Exception {
        RunContext ctx = runContextFactory.of(Map.of());
        var guardrail = guardrail(
            ctx, rule(
                "{{ response.length > 0 and finishReason == 'STOP' and inputTokenCount == 10 and outputTokenCount == 20 }}",
                "violation"
            )
        );

        assertThat(guardrail.checkOrNull(response("Hello", FinishReason.STOP, new TokenUsage(10, 20)))).isNull();
    }
}
