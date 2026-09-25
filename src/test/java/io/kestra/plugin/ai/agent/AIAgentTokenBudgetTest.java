package io.kestra.plugin.ai.agent;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.parallel.ResourceLock;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.IdUtils;
import io.kestra.plugin.ai.MockOpenAI;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.domain.ToolProvider;
import io.kestra.plugin.ai.provider.OpenAI;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.service.tool.ToolExecutor;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class AIAgentTokenBudgetTest {
    @RegisterExtension
    static final MockOpenAI llm = new MockOpenAI();

    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void stopsTheToolLoopWhenTheRenderedTokenBudgetIsExceeded() {
        llm.callTool("fake_tool", "{}");
        var runContext = runContextFactory.of(Map.of("tokenBudget", 30));
        var tool = new CountingToolProvider();

        assertThatThrownBy(() -> agent(Property.ofExpression("{{ tokenBudget }}"), tool).run(runContext))
            .isInstanceOf(IllegalStateException.class)
            .hasMessage(
                "Cumulative token budget exceeded: consumed 40 tokens, configured `maxCumulativeTokens` is 30. " +
                    "Increase `maxCumulativeTokens`, reduce the prompt or tool outputs, or limit `maxSequentialToolsInvocations`."
            );

        assertThat(tool.invocationCount).hasValue(1);
        assertMetric(runContext, "input.token.count", 20);
        assertMetric(runContext, "output.token.count", 20);
        assertMetric(runContext, "total.token.count", 40);
    }

    @Test
    void completesTheToolLoopWhenTheTokenBudgetIsNotExceeded() throws Exception {
        llm.callTool("fake_tool", "{}");
        var runContext = runContextFactory.of(Map.of("tokenBudget", 50));
        var tool = new CountingToolProvider();

        var output = agent(Property.ofExpression("{{ tokenBudget }}"), tool).run(runContext);

        assertThat(output.getTextOutput()).isEqualTo("tool result");
        assertThat(tool.invocationCount).hasValue(1);
        assertMetric(runContext, "input.token.count", 20);
        assertMetric(runContext, "output.token.count", 20);
        assertMetric(runContext, "total.token.count", 40);
    }

    private static AIAgent agent(Property<Integer> tokenBudget, ToolProvider tool) {
        return AIAgent.builder()
            .id(IdUtils.create())
            .provider(
                OpenAI.builder()
                    .type(OpenAI.class.getName())
                    .apiKey(Property.ofValue("demo"))
                    .modelName(Property.ofValue("gpt-4o-mini"))
                    .baseUrl(Property.ofValue(llm.baseUrl()))
                    .build()
            )
            .prompt(Property.ofValue("Use the fake tool."))
            .tools(List.of(tool))
            .configuration(
                ChatConfiguration.builder()
                    .maxCumulativeTokens(tokenBudget)
                    .build()
            )
            .build();
    }

    private static void assertMetric(RunContext runContext, String name, long value) {
        var values = runContext.metrics().stream()
            .filter(metric -> metric.getName().equals(name))
            .map(metric -> ((Number) metric.getValue()).doubleValue())
            .toList();

        assertThat(values).containsExactly((double) value);
    }

    private static final class CountingToolProvider extends ToolProvider {
        private final AtomicInteger invocationCount = new AtomicInteger();

        @Override
        public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) {
            var specification = ToolSpecification.builder()
                .name("fake_tool")
                .description("Returns a deterministic result")
                .build();

            return Map.of(specification, (request, memoryId) ->
            {
                invocationCount.incrementAndGet();
                return "tool result";
            });
        }
    }
}
