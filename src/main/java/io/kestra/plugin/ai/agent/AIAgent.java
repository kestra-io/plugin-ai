package io.kestra.plugin.ai.agent;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.OutputFilesInterface;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.models.tasks.runners.ScriptService;
import io.kestra.core.runners.FilesService;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.ListUtils;
import io.kestra.plugin.ai.AIUtils;
import io.kestra.plugin.ai.domain.*;
import io.kestra.plugin.ai.provider.TimingChatModelListener;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;

import java.net.URI;
import java.util.*;

import static io.kestra.core.utils.Rethrow.throwFunction;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Call an AI agent.",
    description = """
        AI agents are autonomous systems that use a Large Language Model (LLM) to act on a user inputs.
        They can use tools, content retrievers, and be configured with a memory for enhanced context."""
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = """
                Call an AI agent to summarize some information""",
            code = """
                id: ai-agent
                namespace: company.team

                inputs:
                  - id: text
                    type: STRING

                tasks:
                  - id: ai-agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    systemMessage: You are an assistant, can you summarize the text from the user message.
                    prompt: "{{inputs.text}}\""""
        ),
        @Example(
            full = true,
            title = """
                Call an AI agent to with a MCP tool""",
            code = """
                id: ai-agent
                namespace: company.team

                inputs:
                  - id: text
                    type: STRING

                tasks:
                  - id: ai-agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    systemMessage: You are an assistant, can you summarize the text from the user message.
                    prompt: "{{inputs.text}}\""""
        ),
        @Example(
            full = true,
            title = """
                Call an AI agent to with a memory""",
            code = """
                id: ai-agent
                namespace: company.team

                tasks:
                  - id: ai-agent-first
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    memory:
                      type: io.kestra.plugin.ai.memory.KestraKVStore
                    prompt: "Hello, my name is John"
                  - id: ai-agent-second
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    memory:
                      type: io.kestra.plugin.ai.memory.KestraKVStore
                    prompt: "What's my name?\""""
        ),
        @Example(
            full = true,
            title = """
                Call an AI agent to with a content retriever""",
            code = """
                id: ai-agent
                namespace: company.team

                inputs:
                  - id: text
                    type: STRING

                tasks:
                  - id: ai-agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.TavilyWebSearch
                        apiKey: "{{ secret('TAVILY_API_KEY') }}"
                        maxResults: 5
                    prompt: "{{inputs.text}}\""""
        ),
        @Example(
            full = true,
            title = """
                Extract structured outputs with a JSON schema.
                Note that not all model providers support JSON schema, if not, you have to specify the schema inside the prompt.""",
            code = """
                id: structured-output
                namespace: company.team

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: |
                      Hello, my name is John. I was born on January 1, 2000.

                tasks:
                  - id: ai-agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    configuration:
                      responseFormat:
                        type: JSON
                        jsonSchema:
                          type: object
                          properties:
                            name:
                              type: string
                            birth:
                              type: string
                    prompt: "{{inputs.prompt}}"
                """
        ),
        @Example(
            full = true,
            title = """
                Extract output files generated by a tool.
                The tool must generate the file inside the task working directory and the file must be declared inside the task `outputFiles` property.
                You can use the `{{workingDir}}` expression to access the task working directory from within your flow definition.""",
            code = """
                id: output-files
                namespace: company.team

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: |
                      Create a file 'hello.txt' with the content "Hello World"

                tasks:
                  - id: ai-agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    tools:
                    # As the Stdio MCP client uses a docker container,
                    # you must mount the path of the container to the task working directory to be able to access the generated file
                    - type: io.kestra.plugin.ai.tool.StdioMcpClient
                      command: ["docker", "run", "--rm", "-i", "-v", "{{workingDir}}:/tmp", "mcp/filesystem", "/tmp"]
                    outputFiles:
                    - hello.txt
                    prompt: "{{inputs.prompt}}"
                """
        )
    }
)
public class AIAgent extends Task implements RunnableTask<AIOutput>, OutputFilesInterface {
    @Schema(title = "System message", description = "The system message for the language model")
    protected Property<String> systemMessage;

    @Schema(title = "Text prompt", description = "The input prompt for the language model")
    @NotNull
    protected Property<String> prompt;

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

    @Schema(title = "Max sequential tools invocations")
    private Property<Integer> maxSequentialToolsInvocations;

    @Schema(
        title = "Content retrievers",
        description = "Some content retrievers, like WebSearch, can also be used as tools. However, when configured as content retrievers, they will always be used, whereas tools are only invoked when the LLM decides to use them."
    )
    private Property<List<ContentRetrieverProvider>> contentRetrievers;

    @Schema(
        title = "Agent Memory",
        description = "Agent memory will store messages and add them as history inside the LLM context."
    )
    private MemoryProvider memory;

    private Property<List<String>> outputFiles;

    @Override
    public AIOutput run(RunContext runContext) throws Exception {
        Map<String, Object> additionalVariables = outputFiles != null ? Map.of(ScriptService.VAR_WORKING_DIR, runContext.workingDir().path(true).toString()) : Collections.emptyMap();
        List<ToolProvider> toolProviders = ListUtils.emptyOnNull(tools);

        try {
            AiServices<Agent> agent = AiServices.builder(Agent.class)
                .chatModel(provider.chatModel(runContext, configuration))
                .tools(AIUtils.buildTools(runContext, additionalVariables, toolProviders))
                .maxSequentialToolsInvocations(runContext.render(maxSequentialToolsInvocations).as(Integer.class).orElse(Integer.MAX_VALUE))
                .systemMessageProvider(throwFunction(memoryId -> runContext.render(systemMessage).as(String.class).orElse(null)));

            if (memory != null) {
                agent.chatMemory(memory.chatMemory(runContext));
            }

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

            String renderedPrompt = runContext.render(prompt).as(String.class, additionalVariables).orElseThrow();
            Result<AiMessage> completion = agent.build().invoke(renderedPrompt);
            runContext.logger().debug("Generated Completion: {}", completion.content());

            // send metrics for token usage
            TokenUsage tokenUsage = TokenUsage.from(completion.tokenUsage());
            AIUtils.sendMetrics(runContext, tokenUsage);

            return AIOutput.builderFrom(runContext, completion, configuration.computeResponseFormat(runContext).type())
                .outputFiles(gatherOutputFiles(runContext))
                .build();
        } finally {
            toolProviders.forEach(tool -> tool.close(runContext));

            if (memory != null) {
                memory.close(runContext);
            }

            TimingChatModelListener.clear();
        }
    }

    // output files should all be inside the working directory
    private Map<String, URI> gatherOutputFiles(RunContext runContext) throws Exception {
        Map<String, URI> outputFiles = new HashMap<>();
        if (this.outputFiles != null) {
            outputFiles.putAll(FilesService.outputFiles(runContext, runContext.render(this.outputFiles).asList(String.class)));
        }
        return outputFiles;
    }

    interface Agent {
        Result<AiMessage> invoke(String userMessage);
    }

}