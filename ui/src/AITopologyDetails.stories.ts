import type { Meta, StoryObj } from "@storybook/vue3";
import AITopologyDetails from "./components/AITopologyDetails.vue";

const meta: Meta<typeof AITopologyDetails> = {
    title: "Plugin UI / topology-details / AITopologyDetails",
    component: AITopologyDetails,
    tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof AITopologyDetails>;

const openAIProvider = {
    type: "io.kestra.plugin.ai.provider.OpenAI",
    modelName: "gpt-4o",
    apiKey: "{{ secret('OPENAI_API_KEY') }}",
};

const agentTask = {
    id: "ai-agent",
    type: "io.kestra.plugin.ai.agent.AIAgent",
    provider: openAIProvider,
    systemPrompt: "You are a helpful assistant that answers questions about Kestra workflows.",
    tools: [
        { type: "io.kestra.plugin.ai.tool.KestraFlow" },
        { type: "io.kestra.plugin.ai.tool.TavilyWebSearch" },
    ],
};

const chatTask = {
    id: "chat-completion",
    type: "io.kestra.plugin.ai.completion.ChatCompletion",
    provider: openAIProvider,
};

const ragTask = {
    id: "rag-chat",
    type: "io.kestra.plugin.ai.rag.ChatCompletion",
    provider: openAIProvider,
    contentRetriever: { type: "io.kestra.plugin.ai.retriever.EmbeddingStoreRetriever" },
};

export const PreExecution: Story = {
    name: "Pre-execution (AIAgent with tools + system prompt)",
    args: {
        task: agentTask,
        namespace: "company.team",
        flowId: "ai-pipeline",
    },
};

export const ChatCompletionPostExecution: Story = {
    name: "Chat completion post-execution (STOP, token usage)",
    args: {
        task: chatTask,
        namespace: "company.team",
        flowId: "ai-pipeline",
        execution: {
            id: "exec-chat-001",
            namespace: "company.team",
            flowId: "ai-pipeline",
            state: { current: "SUCCESS", startDate: "2024-01-15T10:00:00Z" },
            taskRunList: [
                {
                    id: "tr-001",
                    taskId: "chat-completion",
                    executionId: "exec-chat-001",
                    outputs: {
                        textOutput:
                            "Kestra is an open-source orchestration platform that lets you build, schedule, and monitor workflows as code.",
                        finishReason: "STOP",
                        tokenUsage: {
                            inputTokenCount: 42,
                            outputTokenCount: 31,
                            totalTokenCount: 73,
                        },
                        toolExecutions: [],
                        intermediateResponses: [],
                        guardrailViolated: false,
                    },
                },
            ],
        },
    },
};

export const AgentWithToolCalls: Story = {
    name: "Agent with tool calls (full run)",
    args: {
        task: agentTask,
        namespace: "company.team",
        flowId: "ai-pipeline",
        execution: {
            id: "exec-agent-002",
            namespace: "company.team",
            flowId: "ai-pipeline",
            state: { current: "SUCCESS", startDate: "2024-01-16T09:00:00Z" },
            taskRunList: [
                {
                    id: "tr-002",
                    taskId: "ai-agent",
                    executionId: "exec-agent-002",
                    outputs: {
                        textOutput:
                            "I found 3 recent executions of the 'etl-pipeline' flow: two succeeded and one failed due to a timeout.",
                        finishReason: "STOP",
                        tokenUsage: {
                            inputTokenCount: 120,
                            outputTokenCount: 55,
                            totalTokenCount: 175,
                        },
                        toolExecutions: [
                            {
                                requestId: "call-001",
                                requestName: "KestraFlow",
                                requestArguments: {
                                    namespace: "company.team",
                                    flowId: "etl-pipeline",
                                    action: "list_executions",
                                },
                                result: JSON.stringify([
                                    { id: "exec-1", state: "SUCCESS" },
                                    { id: "exec-2", state: "SUCCESS" },
                                    { id: "exec-3", state: "FAILED" },
                                ]),
                            },
                            {
                                requestId: "call-002",
                                requestName: "TavilyWebSearch",
                                requestArguments: {
                                    query: "Kestra workflow timeout configuration",
                                    maxResults: 3,
                                },
                                result: "Top result: https://kestra.io/docs/workflow-components/tasks#timeout",
                            },
                        ],
                        intermediateResponses: [
                            {
                                id: "resp-1",
                                completion: null,
                                finishReason: "TOOL_EXECUTION",
                                toolExecutionRequests: [{ id: "call-001", name: "KestraFlow" }],
                            },
                            {
                                id: "resp-2",
                                completion: null,
                                finishReason: "TOOL_EXECUTION",
                                toolExecutionRequests: [
                                    { id: "call-002", name: "TavilyWebSearch" },
                                ],
                            },
                        ],
                        guardrailViolated: false,
                    },
                },
            ],
        },
    },
};

export const RagCompletion: Story = {
    name: "RAG completion with 3 sources",
    args: {
        task: ragTask,
        namespace: "company.team",
        flowId: "rag-pipeline",
        execution: {
            id: "exec-rag-003",
            namespace: "company.team",
            flowId: "rag-pipeline",
            state: { current: "SUCCESS", startDate: "2024-01-17T08:30:00Z" },
            taskRunList: [
                {
                    id: "tr-003",
                    taskId: "rag-chat",
                    executionId: "exec-rag-003",
                    outputs: {
                        textOutput:
                            "Based on the documentation, you can configure retries using the `retry` property on any task.",
                        finishReason: "STOP",
                        tokenUsage: {
                            inputTokenCount: 250,
                            outputTokenCount: 48,
                            totalTokenCount: 298,
                        },
                        toolExecutions: [],
                        intermediateResponses: [],
                        guardrailViolated: false,
                        sources: [
                            {
                                content:
                                    "The retry property accepts a type field that can be set to constant or exponential.",
                                metadata: {
                                    source: "https://kestra.io/docs/workflow-components/retries",
                                    score: "0.92",
                                },
                            },
                            {
                                content:
                                    "Each retry attempt waits for the configured interval before re-executing the task.",
                                metadata: {
                                    source: "https://kestra.io/docs/workflow-components/retries",
                                    score: "0.87",
                                },
                            },
                            {
                                content:
                                    "You can set maxAttempts to limit the total number of retry attempts.",
                                metadata: {
                                    source: "https://kestra.io/docs/workflow-components/retries",
                                    score: "0.81",
                                },
                            },
                        ],
                    },
                },
            ],
        },
    },
};

export const ThinkingEnabled: Story = {
    name: "Thinking enabled (reasoning chain)",
    args: {
        task: {
            ...agentTask,
            provider: {
                type: "io.kestra.plugin.ai.provider.Anthropic",
                modelName: "claude-3-7-sonnet-latest",
                apiKey: "{{ secret('ANTHROPIC_API_KEY') }}",
                returnThinking: true,
            },
        },
        namespace: "company.team",
        flowId: "ai-pipeline",
        execution: {
            id: "exec-think-004",
            namespace: "company.team",
            flowId: "ai-pipeline",
            state: { current: "SUCCESS", startDate: "2024-01-18T11:00:00Z" },
            taskRunList: [
                {
                    id: "tr-004",
                    taskId: "ai-agent",
                    executionId: "exec-think-004",
                    outputs: {
                        textOutput: "The optimal batch size for this workload is 500 records.",
                        finishReason: "STOP",
                        tokenUsage: {
                            inputTokenCount: 180,
                            outputTokenCount: 92,
                            totalTokenCount: 272,
                        },
                        thinking:
                            "Let me think about this carefully. The user is asking about optimal batch size. " +
                            "I need to consider memory constraints, throughput, and latency trade-offs. " +
                            "For most ETL workloads, a batch size of 500-1000 records balances memory usage " +
                            "against the overhead of API calls. Given the context of a Kestra flow with " +
                            "downstream database writes, I should recommend something conservative. " +
                            "500 records should work well without overwhelming the destination system. " +
                            "I'll also note that this can be tuned based on observed performance metrics.",
                        toolExecutions: [],
                        intermediateResponses: [],
                        guardrailViolated: false,
                    },
                },
            ],
        },
    },
};

export const GuardrailViolation: Story = {
    name: "Guardrail violation",
    args: {
        task: chatTask,
        namespace: "company.team",
        flowId: "ai-pipeline",
        execution: {
            id: "exec-guard-005",
            namespace: "company.team",
            flowId: "ai-pipeline",
            state: { current: "FAILED", startDate: "2024-01-19T14:00:00Z" },
            taskRunList: [
                {
                    id: "tr-005",
                    taskId: "chat-completion",
                    executionId: "exec-guard-005",
                    outputs: {
                        textOutput: null,
                        finishReason: null,
                        tokenUsage: null,
                        toolExecutions: [],
                        guardrailViolated: true,
                        guardrailViolationMessage:
                            "Output contains potentially sensitive personal data — PII guardrail triggered.",
                    },
                },
            ],
        },
    },
};

export const MaxTokensHit: Story = {
    name: "Max tokens hit (LENGTH finish reason)",
    args: {
        task: chatTask,
        namespace: "company.team",
        flowId: "ai-pipeline",
        execution: {
            id: "exec-len-006",
            namespace: "company.team",
            flowId: "ai-pipeline",
            state: { current: "SUCCESS", startDate: "2024-01-20T16:00:00Z" },
            taskRunList: [
                {
                    id: "tr-006",
                    taskId: "chat-completion",
                    executionId: "exec-len-006",
                    outputs: {
                        textOutput:
                            "This is a very long response that was truncated because the model reached the maximum token limit configured for this request. The response continues with more content but was cut off at the limit set by maxOutputTokens...",
                        finishReason: "LENGTH",
                        tokenUsage: {
                            inputTokenCount: 50,
                            outputTokenCount: 512,
                            totalTokenCount: 562,
                        },
                        toolExecutions: [],
                        intermediateResponses: [],
                        guardrailViolated: false,
                    },
                },
            ],
        },
    },
};
