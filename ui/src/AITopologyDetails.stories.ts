import type { Meta, StoryObj } from "@storybook/vue3";
import { client } from "@kestra-io/kestra-sdk/client";
import AITopologyDetails from "./components/AITopologyDetails.vue";

// ── Mock the POST /expressions/render endpoint ──────────────────────────────
// Storybook has no Kestra backend, so we stub the generated SDK client's transport
// to mimic the server-side DisplayExpressionRenderer:
//   - vars.* / flow.* / globals.* and secret() resolve from the flow context;
//   - inputs.* / outputs.* / execution.* resolve only when an execution is present;
//   - env() / kv() are never resolved.
// Resolution is all-or-nothing per string: one unresolvable reference keeps the
// whole template raw (matching Pebble's single-pass render). This lets the two
// "Expression resolution" stories below demonstrate the real pre/post-execution
// behaviour without a live server.
const MOCK_VARS: Record<string, string> = {
    "vars.model": "claude-3-haiku-20240307",
    "vars.persona": "a precise technical assistant",
};
const MOCK_INPUTS: Record<string, string> = {
    "inputs.question": "What is Kestra in one sentence?",
    "inputs.tone": "concise",
};

function mockResolveTemplate(template: string, hasExecution: boolean): string {
    let resolvable = true;
    const out = template.replace(/\{\{\s*(.*?)\s*}}/g, (rawMatch, exprSource: string) => {
        const expr = exprSource.trim();
        if (expr in MOCK_VARS) return MOCK_VARS[expr];
        const secret = expr.match(/^secret\(\s*['"]([^'"]+)['"]\s*\)$/);
        if (secret) return `[secret: ${secret[1]}]`;
        if (hasExecution && expr in MOCK_INPUTS) return MOCK_INPUTS[expr];
        // env(), kv(), or an execution-scoped reference with no execution → unresolvable.
        resolvable = false;
        return rawMatch;
    });
    // All-or-nothing: keep the original template if anything could not be resolved.
    return resolvable ? out : template;
}

// The generated client invokes `axios(config)`; replace that transport with a stub
// that renders the request's expressions per the rules above.
client.setConfig({
    axios: (async (config: { data?: unknown }) => {
        const body = (typeof config.data === "string" ? JSON.parse(config.data) : config.data) as {
            expressions?: string[];
            executionId?: string;
        };
        const hasExecution = Boolean(body?.executionId);
        const rendered: Record<string, string> = {};
        for (const expression of body?.expressions ?? []) {
            rendered[expression] = mockResolveTemplate(expression, hasExecution);
        }
        return { data: { rendered }, status: 200, statusText: "OK", headers: {}, config };
    }) as never,
});

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
    prompt: "List the last 3 executions of the etl-pipeline flow.",
    tools: [
        { type: "io.kestra.plugin.ai.tool.KestraFlow" },
        { type: "io.kestra.plugin.ai.tool.TavilyWebSearch" },
    ],
};

const chatTask = {
    id: "chat-completion",
    type: "io.kestra.plugin.ai.completion.ChatCompletion",
    provider: openAIProvider,
    systemPrompt: "You are a helpful assistant.\nAnswer concisely in one sentence.",
    prompt: "What is Kestra?",
};

const ragTask = {
    id: "rag-chat",
    type: "io.kestra.plugin.ai.rag.ChatCompletion",
    chatProvider: openAIProvider,
    contentRetriever: { type: "io.kestra.plugin.ai.retriever.EmbeddingStoreRetriever" },
};

// Task whose display-facing fields all carry Pebble expressions from different context
// sources, so the same task renders differently before vs. after an execution exists.
const expressionAgentTask = {
    id: "ai-agent",
    type: "io.kestra.plugin.ai.agent.AIAgent",
    provider: {
        type: "io.kestra.plugin.ai.provider.Anthropic",
        modelName: "{{ vars.model }}", // flow context → resolves pre-execution
        apiKey: "{{ secret('ANTHROPIC_API_KEY') }}",
    },
    // vars + secret → resolves pre-execution (secret masked, never revealed).
    systemPrompt: "You are {{ vars.persona }}.\nAuth token: {{ secret('ANTHROPIC_API_KEY') }}",
    // inputs.* → raw until an execution supplies the value.
    prompt: "{{ inputs.question }}",
};

export const PreExecution: Story = {
    name: "Pre-execution (AIAgent with tools + system prompt)",
    args: {
        task: agentTask,
        namespace: "company.team",
        flowId: "ai-pipeline",
        displayMode: "full",
    },
};

export const ChatCompletionPostExecution: Story = {
    name: "Chat completion post-execution (STOP, token usage)",
    args: {
        task: chatTask,
        namespace: "company.team",
        flowId: "ai-pipeline",
        displayMode: "full",
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
        displayMode: "full",
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
        displayMode: "full",
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
        displayMode: "full",
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
        displayMode: "full",
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

export const ExpressionsPreExecution: Story = {
    name: "Expression resolution — pre-execution (vars + secret resolve, inputs stay raw)",
    args: {
        // No execution: Model resolves to "claude-3-haiku-20240307" and the system message
        // resolves ("...precise technical assistant" + masked secret), but the prompt stays
        // raw as "{{ inputs.question }}" because inputs need an execution.
        task: expressionAgentTask,
        namespace: "company.ai",
        flowId: "ai_pebble_resolution_test",
        displayMode: "full",
    },
};

export const ExpressionsPostExecution: Story = {
    name: "Expression resolution — post-execution (inputs resolve too)",
    args: {
        // With an execution present, the prompt's "{{ inputs.question }}" now also resolves
        // to "What is Kestra in one sentence?" alongside the vars/secret already resolved.
        task: expressionAgentTask,
        namespace: "company.ai",
        flowId: "ai_pebble_resolution_test",
        displayMode: "full",
        execution: {
            id: "exec-expr-001",
            namespace: "company.ai",
            flowId: "ai_pebble_resolution_test",
            state: { current: "SUCCESS", startDate: "2024-01-21T12:00:00Z" },
            taskRunList: [
                {
                    id: "tr-expr-001",
                    taskId: "ai-agent",
                    executionId: "exec-expr-001",
                    outputs: {
                        textOutput:
                            "Kestra is an open-source orchestration platform for building and scheduling workflows as code.",
                        finishReason: "STOP",
                        tokenUsage: {
                            inputTokenCount: 34,
                            outputTokenCount: 18,
                            totalTokenCount: 52,
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

export const MaxTokensHit: Story = {
    name: "Max tokens hit (LENGTH finish reason)",
    args: {
        task: chatTask,
        namespace: "company.team",
        flowId: "ai-pipeline",
        displayMode: "full",
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
