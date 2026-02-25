package io.kestra.plugin.ai.agent;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Metric;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.metrics.Counter;
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
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static io.kestra.core.utils.Rethrow.throwFunction;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Run an AI agent with tools",
    description = """
        Combines a system message, prompt, and optional tools or content retrievers to invoke an LLM and return text/JSON outputs. Content retrievers always run; tools are only called when the model chooses them. `maxSequentialToolsInvocations` defaults to unlimited; memory keeps prior messages, and `outputFiles` collects files from the task working directory."""
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = """
                Summarize arbitrary text with controllable length and language.""",
            code = """
                id: simple_summarizer_agent
                namespace: company.ai

                inputs:
                  - id: summary_length
                    displayName: Summary Length
                    type: SELECT
                    defaults: medium
                    values:
                      - short
                      - medium
                      - long

                  - id: language
                    displayName: Language ISO code
                    type: SELECT
                    defaults: en
                    values:
                      - en
                      - fr
                      - de
                      - es
                      - it
                      - ru
                      - ja

                  - id: text
                    type: STRING
                    displayName: Text to summarize
                    defaults: |
                      Kestra is an open-source orchestration platform that:
                      - Allows you to define workflows declaratively in YAML
                      - Allows non-developers to automate tasks with a no-code interface
                      - Keeps everything versioned and governed, so it stays secure and auditable
                      - Extends easily for custom use cases through plugins and custom scripts.

                      Kestra follows a "start simple and grow as needed" philosophy. You can schedule a basic workflow in a few minutes, then later add Python scripts, Docker containers, or complicated branching logic if the situation calls for it.

                tasks:
                  - id: multilingual_agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    systemMessage: |
                      You are a precise technical assistant.
                      Produce a {{ inputs.summary_length }} summary in {{ inputs.language }}.
                      Keep it factual, remove fluff, and avoid marketing language.
                      If the input is empty or non-text, return a one-sentence explanation.
                      Output format:
                      - 1-2 sentences for 'short'
                      - 2-5 sentences for 'medium'
                      - Up to 5 paragraphs for 'long'
                    prompt: |
                      Summarize the following content: {{ inputs.text }}

                  - id: english_brevity
                    type: io.kestra.plugin.ai.agent.AIAgent
                    prompt: Generate exactly 1 sentence English summary of "{{ outputs.multilingual_agent.textOutput }}"

                pluginDefaults:
                  - type: io.kestra.plugin.ai.agent.AIAgent
                    values:
                      provider:
                        type: io.kestra.plugin.ai.provider.GoogleGemini
                        modelName: gemini-2.5-flash
                        apiKey: "{{ secret('GEMINI_API_KEY') }}"
                """
        ),
        @Example(
            full = true,
            title = """
                Interact with an MCP Server subprocess running in a Docker container""",
            code = """
                id: agent_with_docker_mcp_server_tool
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: What is the current UTC time?

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    prompt: "{{ inputs.prompt }}"
                    provider:
                      type: io.kestra.plugin.ai.provider.OpenAI
                      apiKey: "{{ secret('OPENAI_API_KEY') }}"
                      modelName: gpt-5-nano
                    tools:
                      - type: io.kestra.plugin.ai.tool.DockerMcpClient
                        image: mcp/time
                """
        ),
        @Example(
            full = true,
            title = """
                Run an AI agent with a memory""",
            code = """
                id: agent_with_memory
                namespace: company.ai

                tasks:
                  - id: first_agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    prompt: Hi, my name is John and I live in New York!

                  - id: second_agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    prompt: What's my name and where do I live?

                pluginDefaults:
                  - type: io.kestra.plugin.ai.agent.AIAgent
                    values:
                      provider:
                        type: io.kestra.plugin.ai.provider.OpenAI
                        apiKey: "{{ secret('OPENAI_API_KEY') }}"
                        modelName: gpt-5-mini
                      memory:
                        type: io.kestra.plugin.ai.memory.KestraKVStore
                        memoryId: JOHN
                        ttl: PT1M
                        messages: 5
                """
        ),
        @Example(
            full = true,
            title = """
                Run an AI agent leveraging Tavily Web Search as a content retriever. Note that in contrast to tools, content retrievers are always called to provide context to the prompt, and it's up to the LLM to decide whether to use that retrieved context or not.""",
            code = """
                id: agent_with_content_retriever
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: What is the latest Kestra release and what new features does it include? Name at least 3 new features added exactly in this release.

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    prompt: "{{ inputs.prompt }}"
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-2.5-flash
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.TavilyWebSearch
                        apiKey: "{{ secret('TAVILY_API_KEY') }}"
                """
        ),
        @Example(
            full = true,
            title = """
                Run an AI Agent returning a structured output specified in a JSON schema.
                Note that some providers and models don't support JSON Schema; in those cases, instruct the model to return strict JSON using an inline schema description in the prompt and validate the result downstream.""",
            code = """
                id: agent_with_structured_output
                namespace: company.ai

                inputs:
                  - id: customer_ticket
                    type: STRING
                    defaults: >-
                      I can't log into my account. It says my password is wrong, and the reset link never arrives.

                tasks:
                  - id: support_agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.MistralAI
                      apiKey: "{{ secret('MISTRAL_API_KEY') }}"
                      modelName: open-mistral-7b

                    systemMessage: |
                      You are a classifier that returns ONLY valid JSON matching the schema.
                      Do not add explanations or extra keys.

                    configuration:
                      responseFormat:
                        type: JSON
                        jsonSchema:
                          type: object
                          required: ["category", "priority"]
                          properties:
                            category:
                              type: string
                              enum: ["ACCOUNT", "BILLING", "TECHNICAL", "GENERAL"]
                            priority:
                              type: string
                              enum: ["LOW", "MEDIUM", "HIGH"]

                    prompt: |
                      Classify the following customer message:
                        {{ inputs.customer_ticket }}
                """
        ),

        @Example(
            full = true,
            title = """
                Perform market research with an AI Agent using a web search retriever and save the findings as a Markdown report.
                The retriever gathers up-to-date information, the agent summarizes it, and the filesystem tool writes the result to the task working directory.
                Mount {{workingDir}} to a container path (e.g., /tmp) so the generated report file is accessible and can be collected with `outputFiles`.""",
            code = """
                id: market_research_agent
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: |
                      Research the latest trends in workflow and data orchestration.
                      Use web search to gather current, reliable information from multiple sources.
                      Then create a well-structured Markdown report that includes an introduction,
                      key trends with short explanations, and a conclusion.
                      Save the final report as `report.md` in the `/tmp` directory.

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash
                    prompt: "{{ inputs.prompt }}"
                    systemMessage: |
                      You are a research assistant that must always follow this process:
                      1. Use the TavilyWebSearch content retriever to gather the most relevant and up-to-date information for the user prompt. Do not invent information.
                      2. Summarize and structure the findings clearly in Markdown format. Use headings, bullet points, and links when appropriate.
                      3. Save the final Markdown report as `report.md` in the `/tmp` directory by using the provided filesystem tool.

                      Important rules:
                      - Never output raw text in your response. The final result must always be written to `report.md`.
                      - If no useful results are retrieved, write a short note in `report.md` explaining that no information was found.
                      - Do not attempt to bypass or ignore the retriever or the filesystem tool.

                    contentRetrievers:
                      - type: io.kestra.plugin.ai.retriever.TavilyWebSearch
                        apiKey: "{{ secret('TAVILY_API_KEY') }}"
                        maxResults: 10

                    tools:
                      - type: io.kestra.plugin.ai.tool.DockerMcpClient
                        image: mcp/filesystem
                        command: ["/tmp"]
                        binds: ["{{workingDir}}:/tmp"] # mount host_path:container_path to access the generated report
                    outputFiles:
                      - report.md
                """
        ),

        @Example(
            full = true,
            title = """
                Analyze a numeric series with CodeExecution.
                The agent must call the code tool for all calculations, then explain the results in English.""",
            code = """
                id: agent_with_code_execution_stats
                namespace: company.ai

                inputs:
                  - id: series
                    type: STRING
                    defaults: |
                      12, 15, 15, 18, 21, 99, 102, 102, 104

                tasks:
                  - id: stats_agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash

                    systemMessage: |
                      You are a data analyst.
                      Always use the CodeExecution tool for computations.
                      Then summarize clearly in English.

                    prompt: |
                      Here is a numeric series: {{ inputs.series }}
                      1) Compute mean, median, min, max, and standard deviation.
                      2) Detect outliers using a z-score greater than 2.
                      3) Explain the distribution in 5-8 lines.

                    tools:
                      - type: io.kestra.plugin.ai.tool.CodeExecution
                        apiKey: "{{ secret('RAPID_API_KEY') }}"
                """
        ),
        @Example(
            full = true,
            title = """
                Generate release notes using Google Custom Web Search as a tool.
                Unlike content retrievers, tools are called only when the LLM decides it needs fresh context.""",
            code = """
                id: agent_with_google_custom_search_release_notes
                namespace: company.ai

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: |
                      Find the most recent Kestra release and summarize:
                      - release date
                      - 5 major new features
                      - 3 important bug fixes
                      Answer in English.

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash

                    systemMessage: |
                      You are a release-notes assistant.
                      If you need up-to-date information, call GoogleCustomWebSearch.
                      Summarize sources and avoid hallucinations.

                    prompt: "{{ inputs.prompt }}"

                    tools:
                      - type: io.kestra.plugin.ai.tool.GoogleCustomWebSearch
                        apiKey: "{{ secret('GOOGLE_SEARCH_API_KEY') }}"
                        csi: "{{ secret('GOOGLE_SEARCH_CSI') }}"
                """
        ),
        @Example(
            full = true,
            title = """
                Triage an incident and trigger the right Kestra flow using KestraFlow in implicit mode.
                The agent infers namespace/flowId from the prompt and executes the flow.""",
            code = """
                id: incident_triage_orchestrator
                namespace: company.ai

                inputs:
                  - id: incident
                    type: STRING
                    defaults: |
                      The "billing-prod" SaaS data has been stale for 2 hours.
                      We suspect an API extraction failure from an external provider.

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.OpenAI
                      apiKey: "{{ secret('OPENAI_API_KEY') }}"
                      modelName: gpt-5-mini

                    systemMessage: |
                      You are an incident triage agent.
                      Decide which flow to run to mitigate the issue.
                      Use the kestra_flow tool to trigger it with relevant inputs.

                    prompt: |
                      Incident:
                      {{ inputs.incident }}

                      You can run the following flows in the "prod.ops" namespace:
                      - restart-billing-extract (inputs: service, reason)
                      - run-billing-backfill (inputs: service, sinceHours)
                      - notify-oncall (inputs: team, severity, message)

                      Pick the best flow and execute it using the tool.

                    tools:
                      - type: io.kestra.plugin.ai.tool.KestraFlow
                """
        ),
        @Example(
            full = true,
            title = """
                Route between multiple explicitly-defined Kestra flows.
                Each flow becomes a separate tool and the LLM selects which one to call.""",
            code = """
                id: multi_flow_planner_agent
                namespace: company.ai

                inputs:
                  - id: objective
                    type: SELECT
                    defaults: ingestion
                    values:
                      - ingestion
                      - cleanup
                      - alerting

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash

                    prompt: |
                      User objective: {{ inputs.objective }}
                      Execute the most appropriate flow for this objective.

                    tools:
                      - type: io.kestra.plugin.ai.tool.KestraFlow
                        namespace: prod.data
                        flowId: ingest-daily-snapshots
                        description: Daily ingestion of snapshots

                      - type: io.kestra.plugin.ai.tool.KestraFlow
                        namespace: prod.data
                        flowId: purge-stale-partitions
                        description: Cleanup of obsolete partitions

                      - type: io.kestra.plugin.ai.tool.KestraFlow
                        namespace: prod.ops
                        flowId: send-severity-alert
                        description: Send an on-call alert
                """
        ),
        @Example(
            full = true,
            title = """
                Self-healing automation using KestraTask.
                The agent fills mandatory placeholders ("...") and then runs the task tool.""",
            code = """
                id: agent_using_kestra_task_self_healing
                namespace: company.ai

                inputs:
                  - id: error_message
                    type: STRING
                    defaults: "Disk usage >= 95% on node worker-3"

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash

                    systemMessage: |
                      You are a self-healing automation agent.
                      When remediation is needed, call the KestraTask tool.

                    prompt: |
                      Detected issue: {{ inputs.error_message }}
                      1) Propose a safe remediation action.
                      2) Execute the corresponding task using the tool.

                    tools:
                      - type: io.kestra.plugin.ai.tool.KestraTask
                        tasks:
                          - id: cleanup
                            type: io.kestra.plugin.scripts.shell.Commands
                            commands:
                              - "..."   # Placeholder: the agent will decide real commands.
                            timeout: PT10M
                """
        ),
        @Example(
            full = true,
            title = """
                Find places using an MCP SSE client tool.
                The agent calls the MCP server to retrieve structured results, then ranks them.""",
            code = """
                id: agent_with_sse_mcp_places
                namespace: company.ai

                inputs:
                  - id: city
                    type: STRING
                    defaults: Lyon, France
                  - id: cuisine
                    type: STRING
                    defaults: "bistronomic"

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash

                    systemMessage: |
                      You are a local guide.
                      Use the MCP places tool to search restaurants.
                      Return a short ranked list with brief reasons.

                    prompt: |
                      Find 3 {{ inputs.cuisine }} restaurants in {{ inputs.city }}.
                      Criteria: rating > 4.5, quiet atmosphere, mid-range budget.
                      Provide name, address, and two short reasons for each.

                    tools:
                      - type: io.kestra.plugin.ai.tool.SseMcpClient
                        sseUrl: https://mcp.apify.com/?actors=compass/crawler-google-places
                        timeout: PT3M
                        headers:
                          Authorization: Bearer {{ secret('APIFY_API_TOKEN') }}
                """
        ),
        @Example(
            full = true,
            title = """
                Combine TavilyWebSearch and CodeExecution as tools.
                The agent searches for market data, then computes projections with the code tool.""",
            code = """
                id: agent_research_and_validate_forecast
                namespace: company.ai

                inputs:
                  - id: topic
                    type: STRING
                    defaults: "workflow and data orchestration market"
                  - id: year
                    type: INT
                    defaults: 2028

                tasks:
                  - id: agent
                    type: io.kestra.plugin.ai.agent.AIAgent
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                      modelName: gemini-2.5-flash

                    systemMessage: |
                      You are a market research analyst.
                      1) Use TavilyWebSearch to gather current market size and CAGR.
                      2) Use CodeExecution to project the market size to the target year.
                      3) Summarize in English with sources.

                    prompt: |
                      Topic: {{ inputs.topic }}
                      1) Find credible sources for the current market size and CAGR.
                      2) Project the market size for {{ inputs.year }} using the CAGR.
                      3) Write a compact report (2 paragraphs) plus a list of sources.

                    tools:
                      - type: io.kestra.plugin.ai.tool.TavilyWebSearch
                        apiKey: "{{ secret('TAVILY_API_KEY') }}"
                      - type: io.kestra.plugin.ai.tool.CodeExecution
                        apiKey: "{{ secret('RAPID_API_KEY') }}"
                """
        ),
    },
    metrics = {
        @Metric(
            name = "input.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) input token count"
        ),
        @Metric(
            name = "output.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) output token count"
        ),
        @Metric(
            name = "total.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) total token count"
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

    @Schema(title = "Maximum sequential tools invocations")
    private Property<Integer> maxSequentialToolsInvocations;

    @Schema(
        title = "Content retrievers",
        description = "Some content retrievers, like WebSearch, can also be used as tools. However, when configured as content retrievers, they will always be used, whereas tools are only invoked when the LLM decides to use them."
    )
    private Property<List<ContentRetrieverProvider>> contentRetrievers;

    @Schema(
        title = "Agent memory",
        description = "Agent memory will store messages and add them as history to the LLM context."
    )
    private MemoryProvider memory;

    private Property<List<String>> outputFiles;

    @Override
    public AIOutput run(RunContext runContext) throws Exception {

        Map<String, Object> additionalVariables = outputFiles != null ? Map.of(ScriptService.VAR_WORKING_DIR, runContext.workingDir().path(true).toString()) : Collections.emptyMap();
        String rPrompt = runContext.render(prompt).as(String.class, additionalVariables).orElseThrow();
        List<ToolProvider> toolProviders = ListUtils.emptyOnNull(tools);

        var logger = runContext.logger();

        try {
            AiServices<Agent> agent = AiServices.builder(Agent.class)
                .chatModel(provider.chatModel(runContext, configuration))
                .tools(AIUtils.buildTools(runContext, additionalVariables, toolProviders))
                .maxSequentialToolsInvocations(runContext.render(maxSequentialToolsInvocations).as(Integer.class).orElse(Integer.MAX_VALUE))
                .systemMessageProvider(throwFunction(memoryId -> runContext.render(systemMessage).as(String.class).orElse(null)))
                .toolArgumentsErrorHandler((error, context) -> {
                    logger.error("An error occurred while processing tool arguments for tool {} with request ID {}", context.toolExecutionRequest().name(), context.toolExecutionRequest().id(), error);
                    throw new ToolArgumentsException(error);
                })
                .toolExecutionErrorHandler((error, context) -> {
                    logger.error("An error occurred during tool execution for tool {} with request ID {}", context.toolExecutionRequest().name(), context.toolExecutionRequest().id(), error);
                    throw new ToolExecutionException(error);
                });

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

            Result<AiMessage> completion = agent.build().invoke(rPrompt);
            logger.debug("Generated completion: {}", completion.content());

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

    @Override
    public void kill() {
        if (this.tools != null) {
            this.tools.forEach(tool -> {
                try {
                    tool.kill();
                } catch (Exception ignored) {
                }
            });
        }
    }


    interface Agent {
        Result<AiMessage> invoke(String userMessage);
    }

}
