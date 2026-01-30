package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import dev.langchain4j.service.tool.ToolExecutor;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.core.utils.ListUtils;
import io.kestra.plugin.ai.AIUtils;
import io.kestra.plugin.ai.domain.*;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.util.List;
import java.util.Map;

import static io.kestra.core.utils.Rethrow.throwFunction;

@Getter
@SuperBuilder
@NoArgsConstructor
@Plugin(
    examples =  {
        @Example(
            title = "Call an AI agent as a tool",
            full = true,
            code = {
                """
                    id: ai-agent-with-agent-tools
                    namespace: company.ai

                    inputs:
                      - id: prompt
                        type: STRING
                        defaults: |
                          Each flow can produce outputs that can be consumed by other flows. This is a list property, so that your flow can produce as many outputs as you need.
                          Each output needs to have an ID (the name of the output), a type (the same types you know from inputs, e.g., STRING, URI, or JSON), and a value, which is the actual output value that will be stored in internal storage and passed to other flows when needed.
                    tasks:
                      - id: ai-agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-2.5-flash
                          apiKey: "{{ kv('GEMINI_API_KEY') }}"
                        systemMessage: Summarize the user message, then translate it into French using the provided tool.
                        prompt: "{{inputs.prompt}}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.AIAgent
                            description: Translation expert
                            systemMessage: You are an expert in translating text between multiple languages
                            provider:
                              type: io.kestra.plugin.ai.provider.GoogleGemini
                              modelName: gemini-2.5-flash-lite
                              apiKey: "{{ kv('GEMINI_API_KEY') }}\""""
            }
        ),
    }
)
@JsonDeserialize
@Schema(
    title = "Expose a nested AI Agent as a tool",
    description = """
        Wraps another AI Agent so the parent agent can invoke it as a tool. Provide a unique `name` and `description` per tool; the name defaults to `tool`. Content retrievers configured here always run, while other tools are invoked only when the LLM selects them."""
)
public class AIAgent extends ToolProvider {
    private static final String TOOL_DESCRIPTION = "This tool allows to call an AI agent named '%s'.";

    @Schema(
        title = "Agent name",
        description = "It must be set to a different value than the default in case you want to have multiple agents used as tools in the same task."
    )
    @NotNull
    @Builder.Default
    protected Property<String> name = Property.ofValue("tool");

    @Schema(
        title = "Agent description",
        description = "The description will be used to instruct the LLM what the tool is doing."
    )
    @NotNull
    protected Property<String> description;

    @Schema(title = "System message", description = "The system message for the language model")
    protected Property<String> systemMessage;

    @Schema(title = "Language model provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @Schema(title = "Language model configuration")
    @NotNull
    @PluginProperty
    @Builder.Default
    private ChatConfiguration configuration = ChatConfiguration.empty();

    @Schema(title = "Tools that the LLM may use to augment its response")
    private List<ToolProvider> tools;

    @Schema(title = "Maximum sequential tools invocations")
    private Property<Integer> maxSequentialToolsInvocations;

    @Schema(
        title = "Content retrievers",
        description = "Some content retrievers, like WebSearch, can also be used as tools. However, when configured as content retrievers, they will always be used, whereas tools are only invoked when the LLM decides to use them."
    )
    private Property<List<ContentRetrieverProvider>> contentRetrievers;

    @Getter(AccessLevel.NONE)
    private transient List<ToolProvider> toolProviders;

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws Exception {
        toolProviders = ListUtils.emptyOnNull(tools);

        AiServices<AgentTool> agent = AiServices.builder(AgentTool.class)
            .chatModel(provider.chatModel(runContext, configuration))
            .tools(AIUtils.buildTools(runContext, additionalVariables, toolProviders))
            .maxSequentialToolsInvocations(runContext.render(maxSequentialToolsInvocations).as(Integer.class).orElse(Integer.MAX_VALUE))
            .systemMessageProvider(throwFunction(memoryId -> runContext.render(systemMessage).as(String.class).orElse(null)))
            .toolArgumentsErrorHandler((error, context) -> {
                runContext.logger().error("An error occurred while processing tool arguments for tool {} with request ID {}", context.toolExecutionRequest().name(), context.toolExecutionRequest().id(), error);
                throw new ToolArgumentsException(error);
            })
            .toolExecutionErrorHandler((error, context) -> {
                runContext.logger().error("An error occurred during tool execution for tool {} with request ID {}", context.toolExecutionRequest().name(), context.toolExecutionRequest().id(), error);
                throw new ToolExecutionException(error);
            });

        List<ContentRetriever> toolContentRetrievers = runContext.render(contentRetrievers).asList(ContentRetrieverProvider.class).stream()
            .map(throwFunction(provider -> provider.contentRetriever(runContext)))
            .toList();
        if (!toolContentRetrievers.isEmpty()) {
            QueryRouter queryRouter = new DefaultQueryRouter(toolContentRetrievers.toArray(new ContentRetriever[0]));

            // Create a query router that will route each query to the content retrievers
            agent.retrievalAugmentor(DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build());
        }

        var jsonSchema = JsonObjectSchema.builder()
            .description(runContext.render(this.description).as(String.class).orElseThrow())
            .addStringProperty("prompt", "The AI agent prompt also called a user message")
            .build();

        var rName = runContext.render(this.name).as(String.class).orElseThrow();
        return Map.of(
            ToolSpecification.builder()
                .name("kestra_agent_" + rName)
                .description(TOOL_DESCRIPTION.formatted(rName))
                .parameters(jsonSchema)
                .build(),
            new AgentToolExecutor(runContext, agent.build())
        );
    }

    @Override
    public void close(RunContext runContext) {
        toolProviders.forEach(tool -> tool.close(runContext));
    }

    @Override
    public void kill() {
        if (toolProviders != null) {
            toolProviders.forEach(ToolProvider::kill);
        }
    }

    interface AgentTool {
        Result<String> invoke(String userMessage);
    }

    private static class AgentToolExecutor implements ToolExecutor {
        private final RunContext runContext;
        private final AgentTool agent;

        AgentToolExecutor(RunContext runContext, AgentTool agent) {
            this.runContext = runContext;
            this.agent = agent;
        }

        @Override
        public String execute(ToolExecutionRequest toolExecutionRequest, Object memoryId) {
            runContext.logger().debug("Tool execution request: {}", toolExecutionRequest);
            try {
                var parameters = JacksonMapper.toMap(toolExecutionRequest.arguments());
                var prompt = (String) parameters.get("prompt");

                Result<String> completion = agent.invoke(prompt);
                runContext.logger().debug("Generated completion: {}", completion.content());

                // TODO should we send metrics for token usage
                return completion.content();
            } catch (Exception e) {
                throw new ToolExecutionException(e);
            }
        }
    }
}
