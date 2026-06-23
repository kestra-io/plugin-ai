<script setup lang="ts">
import type { KnownSlotProps } from "@kestra-io/artifact-sdk";
import { KsTopologyDetails, KsTag, KsAlert } from "@kestra-io/design-system";
import { computed, ref, watch, useAttrs, onUnmounted } from "vue";
import { resolveTenant, useClient } from "@kestra-io/kestra-sdk";

const props = defineProps<KnownSlotProps["topology-details"]>();
const attrs = useAttrs();
const isFullView = computed(() => attrs.displayMode === "full");

const taskId = computed(() => props.task?.id as string | undefined);
const taskType = computed(() => props.task?.type as string | undefined);
const isRag = computed(() => taskType.value?.includes(".rag.") ?? false);

// ── Task-level config ────────────────────────────────────────────────────────

function lastSegment(typeStr: string | undefined): string | undefined {
    if (!typeStr) return undefined;
    return typeStr.split(".").at(-1) ?? typeStr;
}

// In post-execution mode the platform may pass a stripped task (id+type only).
// fetchedTaskDef holds the full task definition fetched from the flows API.
const fetchedTaskDef = ref<Record<string, any> | null>(null);
const effectiveTask = computed<Record<string, any>>(() =>
    fetchedTaskDef.value
        ? { ...(props.task as any), ...fetchedTaskDef.value }
        : (props.task as any) ?? {}
);

function findTask(tasks: any[] | undefined, id: string): any {
    if (!tasks?.length) return null;
    for (const t of tasks) {
        if (t.id === id) return t;
        const sub = t.tasks ?? t.errors ?? t.finally;
        if (sub) {
            const found = findTask(Array.isArray(sub) ? sub : [sub], id);
            if (found) return found;
        }
    }
    return null;
}

async function loadFlowTaskDef() {
    if (provider.value) return; // already available from props.task
    const ns = props.namespace as string | undefined;
    const fid = props.flowId as string | undefined;
    const tid = taskId.value;
    if (!ns || !fid || !tid) return;
    if (ns.startsWith("{") || fid.startsWith("{") || tid.startsWith("{")) return;
    const tenant = resolveTenant(undefined);
    if (!tenant || (tenant as string).startsWith("{")) return;
    try {
        const client = useClient();
        const resp = await client.get(
            `/api/v1/${tenant}/flows/${ns}/${fid}`,
            { validateStatus: (s: number) => s === 200 || s === 404 },
        );
        if (resp.status !== 200) return;
        const task = findTask(resp.data?.tasks, tid);
        if (task) fetchedTaskDef.value = task;
    } catch (e: any) {
        console.error("[AITopologyDetails] failed to load flow task def", e);
    }
}

const provider = computed(() => {
    const p = effectiveTask.value[isRag.value ? "chatProvider" : "provider"];
    return lastSegment(p?.type);
});

const modelName = computed(() => {
    const key = isRag.value ? "chatProvider" : "provider";
    return effectiveTask.value[key]?.modelName as string | undefined;
});

const systemMessage = computed(() =>
    (effectiveTask.value.systemPrompt ?? effectiveTask.value.systemMessage) as string | undefined
);

const prompt = computed(() => effectiveTask.value.prompt as string | undefined);

const toolNames = computed<string[]>(() => {
    const tools = effectiveTask.value.tools as any[] | undefined;
    if (!tools?.length) return [];
    return tools.map((t: any) => {
        if (typeof t === "string") return lastSegment(t) ?? t;
        return lastSegment(t?.type) ?? String(t);
    });
});

const toArray = (v: any): any[] => {
    if (Array.isArray(v)) return v;
    if (typeof v === "string") { try { const p = JSON.parse(v); return Array.isArray(p) ? p : []; } catch { return []; } }
    return [];
};

const retrieverNames = computed<string[]>(() => {
    const task = effectiveTask.value;
    const single = task.contentRetriever ?? task.retriever;
    const multi = toArray(task.contentRetrievers ?? task.retrievers);
    const all = single ? [single, ...multi] : multi;
    return all.map((r: any) => lastSegment(r?.type) ?? String(r)).filter(Boolean);
});

const firstRetrieverName = computed(() => retrieverNames.value[0]);

const chatConfigRows = computed(() => {
    const cfg = effectiveTask.value.configuration as Record<string, any> | undefined;
    if (!cfg) return [];
    const LABELS: Record<string, string> = {
        temperature: "Temperature",
        topK: "Top K",
        topP: "Top P",
        seed: "Seed",
        maxToken: "Max tokens",
        thinkingEnabled: "Thinking",
        thinkingBudgetTokens: "Thinking budget",
        returnThinking: "Return thinking",
        logRequests: "Log requests",
        logResponses: "Log responses",
    };
    return Object.entries(LABELS)
        .filter(([k]) => cfg[k] != null)
        .map(([k, label]) => ({ label, value: String(cfg[k]) }));
});

const maxSeqTools = computed(() => {
    const v = effectiveTask.value.maxSequentialToolsInvocations;
    return v != null ? String(v) : undefined;
});

const memoryType = computed(() => lastSegment(effectiveTask.value.memory?.type));

const observabilityType = computed(() => {
    const obs = effectiveTask.value.observability;
    if (!obs) return undefined;
    return lastSegment(obs.type) ?? (typeof obs === "object" ? Object.keys(obs)[0] : undefined);
});

const guardrailsInfo = computed(() => {
    const g = effectiveTask.value.guardrails as Record<string, any> | undefined;
    if (!g) return null;
    const inputRules: string[] = toArray(g.inputGuardrails ?? g.input).map((r: any) => lastSegment(r?.type) ?? String(r));
    const outputRules: string[] = toArray(g.outputGuardrails ?? g.output).map((r: any) => lastSegment(r?.type) ?? String(r));
    return { inputRules, outputRules };
});

// ── Summary rows for compact node ────────────────────────────────────────────

const summaryRows = computed(() => {
    const rows: { label: string; value: string }[] = [
        { label: "Provider", value: provider.value ?? "—" },
        { label: "Model", value: modelName.value ?? "—" },
    ];
    if (isRag.value && firstRetrieverName.value) {
        rows.push({ label: "Retriever", value: firstRetrieverName.value });
    }
    return rows;
});

// ── Execution outputs ─────────────────────────────────────────────────────────

const hasExecution = computed(() => !!props.execution?.id);
const executionId = computed(() => props.execution?.id as string | undefined);

const taskRun = computed(() => {
    const list = props.execution?.taskRunList as any[] | undefined;
    return list?.filter((tr: any) => tr.taskId === taskId.value).at(-1);
});

const fetchedOutputs = ref<Record<string, any> | null>(null);
let currentAbort: AbortController | null = null;

async function loadTaskOutputs(execId: string) {
    if (!execId || execId.startsWith("{")) return;
    const tenant = resolveTenant(undefined);
    if (!tenant || (tenant as string).startsWith("{")) return;

    currentAbort?.abort();
    currentAbort = new AbortController();
    const { signal } = currentAbort;
    fetchedOutputs.value = null;

    try {
        const client = useClient();
        let list = props.execution?.taskRunList as any[] | undefined;
        if (!list) {
            const execResp = await client.get(
                `/api/v1/${tenant}/executions/${execId}`,
                { validateStatus: (s: number) => s === 200 || s === 404, signal },
            );
            if (signal.aborted || execResp.status !== 200) return;
            list = execResp.data?.taskRunList as any[] | undefined;
        }
        const tr = list?.filter((r: any) => r.taskId === taskId.value).at(-1);
        if (!tr?.id || signal.aborted) return;
        const resp = await client.get(
            `/api/v1/${tenant}/outputs/${execId}/${tr.id}`,
            { validateStatus: (s: number) => s === 200 || s === 404, signal },
        );
        if (!signal.aborted) {
            fetchedOutputs.value = resp.status === 200 ? (resp.data ?? null) : null;
        }
    } catch (e: any) {
        if (e?.name !== "AbortError" && e?.code !== "ERR_CANCELED") {
            console.error("[AITopologyDetails] failed to load task outputs", e);
        }
    }
}

onUnmounted(() => currentAbort?.abort());

watch(executionId, (id) => {
    if (id) {
        loadTaskOutputs(id);
        loadFlowTaskDef();
    }
}, { immediate: true });

const outputs = computed(() => fetchedOutputs.value ?? taskRun.value?.outputs ?? null);
const textOutput = computed(() => outputs.value?.textOutput as string | undefined);
const jsonOutput = computed(() => outputs.value?.jsonOutput as Record<string, unknown> | undefined);
const finishReason = computed(() => outputs.value?.finishReason as string | undefined);
const guardrailViolated = computed(() => outputs.value?.guardrailViolated === true);
const guardrailMessage = computed(() => outputs.value?.guardrailViolationMessage as string | undefined);
const tokenUsage = computed(() => outputs.value?.tokenUsage as Record<string, number> | undefined);
const toolExecutions = computed<any[]>(() => outputs.value?.toolExecutions ?? []);
const intermediateResponses = computed<any[]>(() => outputs.value?.intermediateResponses ?? []);
const thinking = computed(() => outputs.value?.thinking as string | undefined);
const sources = computed<any[]>(() => outputs.value?.sources ?? []);

const thinkingTruncated = ref(true);
const THINKING_TRUNCATE = 300;
const thinkingDisplay = computed(() => {
    if (!thinking.value) return undefined;
    if (!thinkingTruncated.value || thinking.value.length <= THINKING_TRUNCATE) return thinking.value;
    return thinking.value.slice(0, THINKING_TRUNCATE) + "…";
});

function formatJson(v: unknown): string {
    try { return JSON.stringify(v, null, 2); } catch { return String(v); }
}

const finishReasonType = computed((): "success" | "warning" | "danger" | "info" | "" => {
    switch (finishReason.value) {
        case "STOP": return "success";
        case "LENGTH": return "warning";
        case "CONTENT_FILTER": return "danger";
        case "TOOL_EXECUTION": return "info";
        default: return "";
    }
});

const hasResponse = computed(() => !guardrailViolated.value && (!!textOutput.value || !!jsonOutput.value));

// ── Cost estimate ─────────────────────────────────────────────────────────────
// USD per 1M tokens — snapshot prices, approximate only

const PRICING: Record<string, { input: number; output: number }> = {
    // OpenAI
    "gpt-4o": { input: 2.5, output: 10 },
    "gpt-4o-mini": { input: 0.15, output: 0.6 },
    "gpt-4-turbo": { input: 10, output: 30 },
    "gpt-3.5-turbo": { input: 0.5, output: 1.5 },
    "o1": { input: 15, output: 60 },
    "o3-mini": { input: 1.1, output: 4.4 },
    // Anthropic
    "claude-opus-4-8": { input: 15, output: 75 },
    "claude-sonnet-4-6": { input: 3, output: 15 },
    "claude-sonnet-4-5": { input: 3, output: 15 },
    "claude-haiku-4-5-20251001": { input: 0.8, output: 4 },
    "claude-3-5-sonnet-20241022": { input: 3, output: 15 },
    "claude-3-7-sonnet-latest": { input: 3, output: 15 },
    "claude-3-opus-20240229": { input: 15, output: 75 },
    "claude-3-haiku-20240307": { input: 0.25, output: 1.25 },
    // Google
    "gemini-2.5-pro": { input: 1.25, output: 10 },
    "gemini-2.5-flash": { input: 0.3, output: 2.5 },
    "gemini-2.0-flash": { input: 0.1, output: 0.4 },
    "gemini-1.5-pro": { input: 1.25, output: 5 },
    "gemini-1.5-flash": { input: 0.075, output: 0.3 },
    "gemini-embedding-001": { input: 0.0, output: 0.0 },
    // Mistral
    "mistral-large-latest": { input: 2, output: 6 },
    "mistral-small-latest": { input: 0.2, output: 0.6 },
    "open-mistral-7b": { input: 0.25, output: 0.25 },
    "open-mixtral-8x7b": { input: 0.7, output: 0.7 },
    // DeepSeek
    "deepseek-chat": { input: 0.27, output: 1.1 },
    "deepseek-reasoner": { input: 0.55, output: 2.19 },
};

const costEstimate = computed(() => {
    if (!tokenUsage.value || !modelName.value) return null;
    const p = PRICING[modelName.value];
    if (!p) return null;
    const total =
        (tokenUsage.value.inputTokenCount ?? 0) / 1_000_000 * p.input +
        (tokenUsage.value.outputTokenCount ?? 0) / 1_000_000 * p.output;
    return total < 0.000001 ? "<$0.000001" : `~$${total.toFixed(6)}`;
});

const tokenMax = computed(() =>
    Math.max(tokenUsage.value?.inputTokenCount ?? 0, tokenUsage.value?.outputTokenCount ?? 0, 1)
);
const tokenInputPct = computed(() =>
    Math.round(((tokenUsage.value?.inputTokenCount ?? 0) / tokenMax.value) * 100)
);
const tokenOutputPct = computed(() =>
    Math.round(((tokenUsage.value?.outputTokenCount ?? 0) / tokenMax.value) * 100)
);
</script>

<template>
    <div class="ai-details">
        <!-- Provider + model always shown in compact node -->
        <KsTopologyDetails :rows="summaryRows" />

        <!-- Everything below: full view (modal / drawer) only -->
        <template v-if="isFullView">

            <!-- ── System message ── -->
            <details v-if="systemMessage" class="ai-section" open>
                <summary class="ai-section__title">System message</summary>
                <pre class="ai-pre">{{ systemMessage }}</pre>
            </details>

            <!-- ── Prompt ── -->
            <details v-if="prompt" class="ai-section" open>
                <summary class="ai-section__title">Prompt</summary>
                <pre class="ai-pre">{{ prompt }}</pre>
            </details>

            <!-- ── Chat Configuration ── -->
            <details v-if="chatConfigRows.length > 0 || maxSeqTools" class="ai-accordion">
                <summary class="ai-accordion__title">Chat Configuration</summary>
                <div class="ai-kv">
                    <template v-for="row in chatConfigRows" :key="row.label">
                        <span class="ai-kv__key">{{ row.label }}</span>
                        <span class="ai-kv__val">{{ row.value }}</span>
                    </template>
                    <template v-if="maxSeqTools">
                        <span class="ai-kv__key">Max tool calls</span>
                        <span class="ai-kv__val">{{ maxSeqTools }}</span>
                    </template>
                </div>
            </details>

            <!-- ── Content Retrievers ── -->
            <details v-if="retrieverNames.length > 0" class="ai-accordion">
                <summary class="ai-accordion__title">Content Retrievers</summary>
                <div class="ai-tags ai-tags--gap">
                    <KsTag v-for="name in retrieverNames" :key="name" size="small" type="info">{{ name }}</KsTag>
                </div>
            </details>

            <!-- ── Memory ── -->
            <details v-if="memoryType" class="ai-accordion">
                <summary class="ai-accordion__title">Memory</summary>
                <div class="ai-kv">
                    <span class="ai-kv__key">Type</span>
                    <span class="ai-kv__val">{{ memoryType }}</span>
                </div>
            </details>

            <!-- ── Observability ── -->
            <details v-if="observabilityType" class="ai-accordion">
                <summary class="ai-accordion__title">Observability</summary>
                <div class="ai-kv">
                    <span class="ai-kv__key">Provider</span>
                    <span class="ai-kv__val">{{ observabilityType }}</span>
                </div>
            </details>

            <!-- ── Guardrails ── -->
            <details v-if="guardrailsInfo" class="ai-accordion">
                <summary class="ai-accordion__title">Guardrails</summary>
                <div class="ai-kv">
                    <template v-if="guardrailsInfo.inputRules.length > 0">
                        <span class="ai-kv__key">Input</span>
                        <span class="ai-kv__val">{{ guardrailsInfo.inputRules.join(", ") }}</span>
                    </template>
                    <template v-if="guardrailsInfo.outputRules.length > 0">
                        <span class="ai-kv__key">Output</span>
                        <span class="ai-kv__val">{{ guardrailsInfo.outputRules.join(", ") }}</span>
                    </template>
                </div>
            </details>

            <!-- ── Execution outputs ── -->
            <template v-if="hasExecution && outputs">
                <KsAlert
                    v-if="guardrailViolated"
                    type="error"
                    :title="guardrailMessage ?? 'Guardrail violated'"
                    show-icon
                    class="ai-alert"
                />

                <details v-if="hasResponse || finishReason || tokenUsage" class="ai-section" open>
                    <summary class="ai-section__title">
                        Answer
                        <KsTag v-if="finishReason" :type="finishReasonType" size="small">{{ finishReason }}</KsTag>
                    </summary>

                    <template v-if="hasResponse">
                        <pre v-if="textOutput" class="ai-pre">{{ textOutput }}</pre>
                        <pre v-else-if="jsonOutput" class="ai-pre ai-pre--json">{{ formatJson(jsonOutput) }}</pre>
                    </template>

                    <div v-if="tokenUsage" class="ai-tokens">
                        <div class="ai-token-bar">
                            <span class="ai-token-bar__label">Input</span>
                            <div class="ai-token-bar__track">
                                <div class="ai-token-bar__fill ai-token-bar__fill--input" :style="{ width: tokenInputPct + '%' }"></div>
                            </div>
                            <span class="ai-token-bar__count">{{ tokenUsage.inputTokenCount ?? "—" }}</span>
                        </div>
                        <div class="ai-token-bar">
                            <span class="ai-token-bar__label">Output</span>
                            <div class="ai-token-bar__track">
                                <div class="ai-token-bar__fill ai-token-bar__fill--output" :style="{ width: tokenOutputPct + '%' }"></div>
                            </div>
                            <span class="ai-token-bar__count">{{ tokenUsage.outputTokenCount ?? "—" }}</span>
                        </div>
                        <div class="ai-token-summary">
                            <div class="ai-token-box">
                                <span class="ai-token-box__label">Total tokens</span>
                                <span class="ai-token-box__value">{{ tokenUsage.totalTokenCount ?? "—" }}</span>
                            </div>
                            <div v-if="costEstimate" class="ai-token-box">
                                <span class="ai-token-box__label">Cost est.</span>
                                <span class="ai-token-box__value">{{ costEstimate }}</span>
                            </div>
                        </div>
                    </div>
                </details>

                <!-- Tool call timeline -->
                <details v-if="toolExecutions.length > 0" class="ai-accordion">
                    <summary class="ai-accordion__title">Tool calls ({{ toolExecutions.length }})</summary>
                    <div
                        v-for="(exec, i) in toolExecutions"
                        :key="exec.requestId ?? i"
                        class="ai-tool-call"
                    >
                        <span class="ai-tool-call__name">{{ exec.requestName }}</span>
                        <pre class="ai-pre ai-pre--json">{{ formatJson(exec.requestArguments) }}</pre>
                        <pre v-if="exec.result" class="ai-pre">{{ exec.result }}</pre>
                    </div>
                </details>

                <!-- Reasoning chain -->
                <details v-if="thinking || intermediateResponses.length > 0" class="ai-accordion">
                    <summary class="ai-accordion__title">Reasoning chain</summary>
                    <template v-if="thinking">
                        <pre class="ai-pre">{{ thinkingDisplay }}</pre>
                        <button
                            v-if="thinking.length > THINKING_TRUNCATE"
                            class="ai-show-more"
                            @click="thinkingTruncated = !thinkingTruncated"
                        >{{ thinkingTruncated ? "Show more" : "Show less" }}</button>
                    </template>
                    <div
                        v-for="(resp, i) in intermediateResponses"
                        :key="resp.id ?? i"
                        class="ai-intermediate"
                    >
                        <span class="ai-intermediate__step">{{ i + 1 }}</span>
                        <span class="ai-intermediate__text">{{ resp.completion }}</span>
                    </div>
                </details>

                <!-- RAG sources -->
                <details v-if="sources.length > 0" class="ai-accordion">
                    <summary class="ai-accordion__title">RAG Sources ({{ sources.length }})</summary>
                    <div v-for="(src, i) in sources" :key="i" class="ai-source">
                        <p class="ai-source__content">{{ src.content }}</p>
                        <dl v-if="src.metadata" class="ai-source__meta">
                            <template v-for="(val, key) in src.metadata" :key="key">
                                <dt>{{ key }}</dt><dd>{{ val }}</dd>
                            </template>
                        </dl>
                    </div>
                </details>
            </template>

        </template>
    </div>
</template>

<style scoped>
.ai-details {
    --ai-font-sm: 0.72rem;
    --ai-gap-xs: 0.25rem;
    --ai-gap-sm: 0.5rem;
    --ai-fw-medium: 500;
    --ai-fw-bold: 700;
    --ai-radius: 4px;
    --ai-color-border: var(--ks-border-subtle, rgba(255,255,255,.1));
    --ai-color-surface: var(--ks-surface-secondary, rgba(255,255,255,.04));
    --ai-color-text-muted: var(--ks-text-secondary, #9ca3af);
    --ai-color-primary: var(--ks-color-primary, #7c3aed);

    padding: 0 0.75rem 0;
    font-size: var(--ai-font-sm);
    line-height: 1.5;
}

/* ── tag groups ─────────────────────────────────────────────────────── */
.ai-tags {
    display: flex;
    flex-wrap: wrap;
    gap: var(--ai-gap-xs);
}
.ai-tags--gap {
    margin-top: 0.35rem;
}

/* ── flat section (system message, prompt, answer) ──────────────────── */
.ai-section {
    border-top: 1px solid var(--ai-color-border);
    margin-top: 0.5rem;
    padding-top: 0.35rem;
}

.ai-section__title {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    font-size: 0.72rem;
    font-weight: var(--ai-fw-bold);
    color: var(--ks-text-primary, inherit);
    padding: 0.1rem 0 0.35rem;
    cursor: pointer;
    user-select: none;
    list-style: none;
}

.ai-section__title::-webkit-details-marker { display: none; }

.ai-section__title::after {
    content: "▲";
    font-size: 0.45rem;
    color: var(--ai-color-text-muted);
    transition: transform 0.15s;
}

details.ai-section:not([open]) > .ai-section__title::after {
    transform: rotate(180deg);
}

/* ── native accordion ───────────────────────────────────────────────── */
.ai-accordion {
    border-top: 1px solid var(--ai-color-border);
    margin-top: 0.5rem;
    padding-top: 0.35rem;
}

.ai-accordion__title {
    font-size: 0.72rem;
    font-weight: var(--ai-fw-bold);
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.3rem;
    user-select: none;
    color: var(--ks-text-primary, inherit);
    padding: 0.1rem 0;
}

.ai-accordion__title::before {
    content: "▶";
    font-size: 0.5rem;
    transition: transform 0.15s;
    flex-shrink: 0;
}

details[open] > .ai-accordion__title::before {
    transform: rotate(90deg);
}

/* hide default marker in Safari / Firefox */
.ai-accordion__title::-webkit-details-marker { display: none; }

/* ── key-value grid ─────────────────────────────────────────────────── */
.ai-kv {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.15rem 0.75rem;
    margin-top: 0.35rem;
    font-size: var(--ai-font-sm);
}

.ai-kv__key {
    color: var(--ai-color-text-muted);
    white-space: nowrap;
}

.ai-kv__val {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── pre blocks ─────────────────────────────────────────────────────── */
.ai-pre {
    margin: var(--ai-gap-xs) 0 0;
    padding: 0.35rem var(--ai-gap-sm);
    border-radius: var(--ai-radius);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.65rem;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--ai-color-surface);
    color: var(--ai-color-text-muted);
    max-height: 10rem;
    overflow-y: auto;
}

.ai-pre--json {
    background: var(--ks-color-surface-code, #1e1e2e);
    color: var(--ks-color-text-code, #cdd6f4);
}

/* ── token progress bars ────────────────────────────────────────────── */
.ai-tokens {
    border-top: 1px solid var(--ai-color-border);
    margin-top: 0.5rem;
    padding-top: 0.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}

.ai-token-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: var(--ai-font-sm);
}

.ai-token-bar__label {
    width: 3.5rem;
    flex-shrink: 0;
    color: var(--ai-color-text-muted);
}

.ai-token-bar__track {
    flex: 1;
    height: 0.45rem;
    border-radius: 99px;
    background: var(--ai-color-surface);
    overflow: hidden;
}

.ai-token-bar__fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.3s ease;
}

.ai-token-bar__fill--input  { background: #7c86ef; }
.ai-token-bar__fill--output { background: #34d399; }

.ai-token-bar__count {
    width: 2.5rem;
    text-align: right;
    flex-shrink: 0;
}

.ai-token-summary {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.35rem;
}

.ai-token-box {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    padding: 0.4rem 0.6rem;
    border-radius: var(--ai-radius);
    background: var(--ai-color-surface);
}

.ai-token-box__label {
    font-size: 0.62rem;
    color: var(--ai-color-text-muted);
}

.ai-token-box__value {
    font-size: 0.9rem;
    font-weight: var(--ai-fw-bold);
}

/* ── tool call entries ──────────────────────────────────────────────── */
.ai-tool-call {
    margin-top: 0.4rem;
    border-left: 2px solid var(--ai-color-primary);
    padding-left: 0.5rem;
}

.ai-tool-call__name {
    font-weight: var(--ai-fw-medium);
    font-size: 0.68rem;
}

/* ── intermediate reasoning steps ──────────────────────────────────── */
.ai-intermediate {
    display: flex;
    gap: 0.4rem;
    align-items: flex-start;
    margin-bottom: var(--ai-gap-xs);
    margin-top: 0.35rem;
}

.ai-intermediate__step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.1rem;
    height: 1.1rem;
    border-radius: 50%;
    background: var(--ai-color-primary);
    color: #fff;
    font-size: 0.6rem;
    font-weight: var(--ai-fw-bold);
    flex-shrink: 0;
}

.ai-intermediate__text {
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── RAG sources ────────────────────────────────────────────────────── */
.ai-source {
    border-left: 2px solid var(--ai-color-primary);
    padding: var(--ai-gap-xs) var(--ai-gap-sm);
    margin-top: 0.35rem;
    font-size: 0.65rem;
}

.ai-source__content {
    margin: 0 0 0.2rem;
    font-style: italic;
}

.ai-source__meta {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.1rem 0.4rem;
    margin: 0;
    color: var(--ai-color-text-muted);
}

.ai-source__meta dt { font-weight: var(--ai-fw-medium); }
.ai-source__meta dd { margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── alert spacing ──────────────────────────────────────────────────── */
.ai-alert { margin-top: 0.5rem; }

/* ── show more button ───────────────────────────────────────────────── */
.ai-show-more {
    all: unset;
    cursor: pointer;
    font-size: 0.65rem;
    color: var(--ai-color-primary);
    text-decoration: underline;
    margin-top: 0.2rem;
    display: block;
}
</style>
