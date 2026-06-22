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

const provider = computed(() => {
    const p = (props.task as any)?.[isRag.value ? "chatProvider" : "provider"];
    return lastSegment(p?.type);
});

const modelName = computed(() => {
    const key = isRag.value ? "chatProvider" : "provider";
    return (props.task as any)?.[key]?.modelName as string | undefined;
});

const systemMessage = computed(() =>
    ((props.task as any)?.systemPrompt ?? (props.task as any)?.systemMessage) as string | undefined
);

const prompt = computed(() => (props.task as any)?.prompt as string | undefined);

const toolNames = computed<string[]>(() => {
    const tools = (props.task as any)?.tools as any[] | undefined;
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
    const task = props.task as any;
    const single = task?.contentRetriever ?? task?.retriever;
    const multi = toArray(task?.contentRetrievers ?? task?.retrievers);
    const all = single ? [single, ...multi] : multi;
    return all.map((r: any) => lastSegment(r?.type) ?? String(r)).filter(Boolean);
});

// Compact topology node shows first retriever in summary rows (RAG)
const firstRetrieverName = computed(() => retrieverNames.value[0]);

// Chat configuration — only non-null entries
const chatConfigRows = computed(() => {
    const cfg = (props.task as any)?.configuration as Record<string, any> | undefined;
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
    const v = (props.task as any)?.maxSequentialToolsInvocations;
    return v != null ? String(v) : undefined;
});

const memoryType = computed(() => lastSegment((props.task as any)?.memory?.type));

const observabilityType = computed(() => {
    const obs = (props.task as any)?.observability;
    if (!obs) return undefined;
    return lastSegment(obs.type) ?? (typeof obs === "object" ? Object.keys(obs)[0] : undefined);
});

const guardrailsInfo = computed(() => {
    const g = (props.task as any)?.guardrails as Record<string, any> | undefined;
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
    if (isFullView.value && toolNames.value.length > 0) {
        rows.push({ label: "Tools", value: toolNames.value.join(", ") });
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

    // validateStatus bypasses the global 404 interceptor (coreStore.error = 404)
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

watch(executionId, (id) => { if (id) loadTaskOutputs(id); }, { immediate: true });

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
</script>

<template>
    <div class="ai-details">
        <!-- Provider + model always shown in compact node -->
        <KsTopologyDetails :rows="summaryRows" />

        <!-- Everything below: full view (modal / drawer) only -->
        <template v-if="isFullView">

            <!-- ── System message ── -->
            <div v-if="systemMessage" class="ai-section">
                <span class="ai-section__label">System message</span>
                <pre class="ai-pre">{{ systemMessage }}</pre>
            </div>

            <!-- ── Prompt ── -->
            <div v-if="prompt" class="ai-section">
                <span class="ai-section__label">Prompt</span>
                <pre class="ai-pre">{{ prompt }}</pre>
            </div>

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

                <template v-if="hasResponse">
                    <p v-if="textOutput" class="ai-response-text">{{ textOutput }}</p>
                    <pre v-else-if="jsonOutput" class="ai-pre ai-pre--json">{{ formatJson(jsonOutput) }}</pre>
                </template>

                <div v-if="finishReason || tokenUsage" class="ai-meta-row">
                    <KsTag v-if="finishReason" :type="finishReasonType" size="small">{{ finishReason }}</KsTag>
                    <span v-if="tokenUsage" class="ai-tokens">
                        <span>in: {{ tokenUsage.inputTokenCount ?? "—" }}</span>
                        <span>out: {{ tokenUsage.outputTokenCount ?? "—" }}</span>
                        <span>total: {{ tokenUsage.totalTokenCount ?? "—" }}</span>
                    </span>
                </div>

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

    position: relative;
    z-index: 1;
    padding: 0.5rem 0.75rem;
    font-size: var(--ai-font-sm);
    line-height: 1.5;
}

/* ── labeled section (Tools / System message / Prompt) ──────────────── */
.ai-section {
    margin-top: 0.6rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.ai-section__label {
    font-size: 0.65rem;
    font-weight: var(--ai-fw-bold);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--ai-color-text-muted);
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

/* ── response text ──────────────────────────────────────────────────── */
.ai-response-text {
    margin: var(--ai-gap-xs) 0 0;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 6rem;
    overflow-y: auto;
}

/* ── meta row (finish reason + tokens) ─────────────────────────────── */
.ai-meta-row {
    display: flex;
    align-items: center;
    gap: var(--ai-gap-sm);
    margin-top: 0.375rem;
    flex-wrap: wrap;
}

.ai-tokens {
    display: flex;
    gap: var(--ai-gap-sm);
    font-size: 0.65rem;
    color: var(--ai-color-text-muted);
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
