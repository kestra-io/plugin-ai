<script setup lang="ts">
import type { TopologyDetailsProps } from "@kestra-io/artifact-sdk";
import { computed, ref, watch, useAttrs } from "vue";
import { execution as fetchExecution } from "@kestra-io/kestra-sdk";

const props = defineProps<TopologyDetailsProps>();
const attrs = useAttrs();
const isFullView = computed(() => attrs.displayMode === "full");

const taskId = computed(() => props.task?.id as string | undefined);
const taskType = computed(() => props.task?.type as string | undefined);
const isRag = computed(() => taskType.value?.includes("rag.") ?? false);

// Pre-execution: extract config from task props directly
const provider = computed(() => {
    const p = (props.task as any)?.provider;
    if (!p) return undefined;
    const typeStr = p.type as string | undefined;
    if (!typeStr) return undefined;
    // Extract last segment: "io.kestra.plugin.ai.provider.OpenAI" → "OpenAI"
    const segments = typeStr.split(".");
    return segments.at(-1) ?? typeStr;
});

const modelName = computed(() => (props.task as any)?.provider?.modelName as string | undefined);
const systemPrompt = computed(() => (props.task as any)?.systemPrompt as string | undefined);

const toolNames = computed<string[]>(() => {
    const tools = (props.task as any)?.tools as any[] | undefined;
    if (!tools?.length) return [];
    return tools.map((t: any) => {
        if (typeof t === "string") {
            const parts = (t as string).split(".");
            return parts.at(-1) ?? t;
        }
        const typeStr = t?.type as string | undefined;
        if (typeStr) {
            const parts = typeStr.split(".");
            return parts.at(-1) ?? typeStr;
        }
        return String(t);
    });
});

const retrieverType = computed(() => {
    if (!isRag.value) return undefined;
    const r = (props.task as any)?.contentRetriever ?? (props.task as any)?.retriever;
    if (!r) return undefined;
    const typeStr = r.type as string | undefined;
    if (!typeStr) return undefined;
    const segments = typeStr.split(".");
    return segments.at(-1) ?? typeStr;
});

// Execution state
const hasExecution = computed(() => !!props.execution?.id);
const executionId = computed(() => props.execution?.id as string | undefined);

const taskRun = computed(() => {
    const list = props.execution?.taskRunList as any[] | undefined;
    return list?.filter((tr: any) => tr.taskId === taskId.value).at(-1);
});

// Fetch full execution to get outputs
const fetchedOutputs = ref<Record<string, any> | null>(null);

async function loadTaskOutputs(execId: string) {
    try {
        const exec = await fetchExecution({ path: { executionId: execId } });
        const list = exec.taskRunList as any[] | undefined;
        const tr = list?.filter((tr: any) => tr.taskId === taskId.value).at(-1);
        fetchedOutputs.value = (tr as any)?.outputs ?? null;
    } catch {
        /* best-effort */
    }
}

watch(
    executionId,
    (id) => {
        if (id) loadTaskOutputs(id);
    },
    { immediate: true },
);

const outputs = computed(() => fetchedOutputs.value ?? taskRun.value?.outputs ?? null);

// Derived output fields
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

// Collapsible states
const systemPromptOpen = ref(false);
const toolCallsOpen = ref<Record<number, boolean>>({});
const thinkingOpen = ref(false);
const sourcesOpen = ref(false);
const thinkingTruncated = ref(true);

const THINKING_TRUNCATE_LENGTH = 300;

const thinkingDisplay = computed(() => {
    if (!thinking.value) return undefined;
    if (!thinkingTruncated.value || thinking.value.length <= THINKING_TRUNCATE_LENGTH) {
        return thinking.value;
    }
    return thinking.value.slice(0, THINKING_TRUNCATE_LENGTH) + "…";
});

function toggleToolCall(index: number) {
    toolCallsOpen.value = { ...toolCallsOpen.value, [index]: !toolCallsOpen.value[index] };
}

function formatJson(v: unknown): string {
    try {
        return JSON.stringify(v, null, 2);
    } catch {
        return String(v);
    }
}

const finishReasonClass = computed(() => {
    switch (finishReason.value) {
        case "STOP":
            return "ai-badge--stop";
        case "LENGTH":
            return "ai-badge--length";
        case "CONTENT_FILTER":
            return "ai-badge--content-filter";
        case "TOOL_EXECUTION":
            return "ai-badge--tool";
        default:
            return "ai-badge--default";
    }
});

const hasResponse = computed(
    () => !guardrailViolated.value && (!!textOutput.value || !!jsonOutput.value),
);
</script>

<template>
    <div class="ai-details">
        <!-- Pre-execution: provider + model always shown -->
        <dl class="ai-grid">
            <dt>Provider</dt>
            <dd>{{ provider ?? "—" }}</dd>
            <dt>Model</dt>
            <dd>{{ modelName ?? "—" }}</dd>
            <template v-if="isRag && retrieverType">
                <dt>Retriever</dt>
                <dd>{{ retrieverType }}</dd>
            </template>
        </dl>

        <!-- System prompt collapsible (pre-execution) -->
        <div v-if="systemPrompt" class="ai-collapsible">
            <button class="ai-collapsible__toggle" @click="systemPromptOpen = !systemPromptOpen">
                <span class="ai-collapsible__icon">{{ systemPromptOpen ? "▾" : "▸" }}</span>
                System prompt
            </button>
            <pre v-if="systemPromptOpen" class="ai-pre ai-pre--muted">{{ systemPrompt }}</pre>
        </div>

        <!-- Tools list (pre-execution) -->
        <div v-if="toolNames.length > 0" class="ai-tools">
            <span class="ai-label">Tools:</span>
            <span v-for="name in toolNames" :key="name" class="ai-pill">{{ name }}</span>
        </div>

        <!-- Post-execution panel -->
        <template v-if="hasExecution && outputs">
            <!-- Guardrail violation — shown prominently, replaces response section -->
            <div v-if="guardrailViolated" class="ai-guardrail">
                <span class="ai-guardrail__icon">⚠</span>
                <div>
                    <div class="ai-guardrail__title">Guardrail violated</div>
                    <div v-if="guardrailMessage" class="ai-guardrail__msg">{{ guardrailMessage }}</div>
                </div>
            </div>

            <!-- LLM response -->
            <template v-if="hasResponse">
                <section v-if="textOutput" class="ai-section">
                    <h4 class="ai-section__title">Response</h4>
                    <p class="ai-response-text">{{ textOutput }}</p>
                </section>
                <section v-else-if="jsonOutput" class="ai-section">
                    <h4 class="ai-section__title">Response</h4>
                    <pre class="ai-pre ai-pre--json">{{ formatJson(jsonOutput) }}</pre>
                </section>
            </template>

            <!-- Finish reason + token usage — compact summary row -->
            <dl v-if="finishReason || tokenUsage" class="ai-grid ai-grid--compact">
                <template v-if="finishReason">
                    <dt>Finish reason</dt>
                    <dd>
                        <span :class="['ai-badge', finishReasonClass]">{{ finishReason }}</span>
                    </dd>
                </template>
                <template v-if="tokenUsage">
                    <dt>Tokens</dt>
                    <dd class="ai-token-row">
                        <span>in: {{ tokenUsage.inputTokenCount ?? "—" }}</span>
                        <span>out: {{ tokenUsage.outputTokenCount ?? "—" }}</span>
                        <span>total: {{ tokenUsage.totalTokenCount ?? "—" }}</span>
                    </dd>
                </template>
            </dl>

            <!-- Full-view extras: tool calls, thinking, RAG sources -->
            <template v-if="isFullView">
                <!-- Tool call timeline -->
                <section v-if="toolExecutions.length > 0" class="ai-section">
                    <h4 class="ai-section__title">Tool Calls</h4>
                    <div
                        v-for="(exec, i) in toolExecutions"
                        :key="exec.requestId ?? i"
                        class="ai-tool-call"
                    >
                        <button class="ai-tool-call__header" @click="toggleToolCall(i)">
                            <span class="ai-tool-call__name">{{ exec.requestName }}</span>
                            <span class="ai-collapsible__icon">{{
                                toolCallsOpen[i] ? "▾" : "▸"
                            }}</span>
                        </button>
                        <template v-if="toolCallsOpen[i]">
                            <div class="ai-tool-call__body">
                                <div class="ai-tool-call__label">Arguments</div>
                                <pre class="ai-pre ai-pre--json">{{
                                    formatJson(exec.requestArguments)
                                }}</pre>
                                <div v-if="exec.result" class="ai-tool-call__label">Result</div>
                                <pre v-if="exec.result" class="ai-pre ai-pre--muted">{{
                                    exec.result
                                }}</pre>
                            </div>
                        </template>
                    </div>
                </section>

                <!-- Thinking / reasoning chain -->
                <section
                    v-if="thinking || intermediateResponses.length > 0"
                    class="ai-section"
                >
                    <h4 class="ai-section__title">
                        <button
                            class="ai-collapsible__toggle"
                            @click="thinkingOpen = !thinkingOpen"
                        >
                            <span class="ai-collapsible__icon">{{
                                thinkingOpen ? "▾" : "▸"
                            }}</span>
                            Reasoning Chain
                        </button>
                    </h4>
                    <template v-if="thinkingOpen">
                        <div v-if="thinking">
                            <pre class="ai-pre ai-pre--muted">{{ thinkingDisplay }}</pre>
                            <button
                                v-if="thinking.length > THINKING_TRUNCATE_LENGTH"
                                class="ai-show-more"
                                @click="thinkingTruncated = !thinkingTruncated"
                            >
                                {{ thinkingTruncated ? "Show more" : "Show less" }}
                            </button>
                        </div>
                        <div
                            v-for="(resp, i) in intermediateResponses"
                            :key="resp.id ?? i"
                            class="ai-intermediate"
                        >
                            <span class="ai-intermediate__step">{{ i + 1 }}</span>
                            <span class="ai-intermediate__text">{{ resp.completion }}</span>
                        </div>
                    </template>
                </section>

                <!-- RAG sources -->
                <section v-if="sources.length > 0" class="ai-section">
                    <h4 class="ai-section__title">
                        <button
                            class="ai-collapsible__toggle"
                            @click="sourcesOpen = !sourcesOpen"
                        >
                            <span class="ai-collapsible__icon">{{
                                sourcesOpen ? "▾" : "▸"
                            }}</span>
                            RAG Sources ({{ sources.length }})
                        </button>
                    </h4>
                    <template v-if="sourcesOpen">
                        <div v-for="(src, i) in sources" :key="i" class="ai-source">
                            <p class="ai-source__content">{{ src.content }}</p>
                            <dl v-if="src.metadata" class="ai-source__meta">
                                <template v-for="(val, key) in src.metadata" :key="key">
                                    <dt>{{ key }}</dt>
                                    <dd>{{ val }}</dd>
                                </template>
                            </dl>
                        </div>
                    </template>
                </section>
            </template>
        </template>
    </div>
</template>

<style scoped>
.ai-details {
    position: relative;
    z-index: 1;
    padding: 0.5rem 0.75rem;
    font-size: 0.7rem;
    line-height: 1.4;
}

.ai-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.15rem 0.625rem;
    margin: 0 0 0.375rem;
}

.ai-grid--compact {
    margin-top: 0.375rem;
}

.ai-grid dt {
    font-weight: 500;
    color: var(--ks-color-text-secondary, #6b7280);
    white-space: nowrap;
}

.ai-grid dd {
    margin: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.ai-section {
    margin-top: 0.5rem;
}

.ai-section__title {
    margin: 0 0 0.25rem;
    font-size: 0.6875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ks-color-text-secondary, #6b7280);
}

.ai-label {
    font-weight: 500;
    color: var(--ks-color-text-secondary, #6b7280);
    margin-right: 0.25rem;
}

.ai-tools {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.2rem;
    margin-bottom: 0.375rem;
}

.ai-pill {
    display: inline-block;
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
    background: var(--ks-color-surface-subtle, #f3f4f6);
    color: var(--ks-color-text-secondary, #6b7280);
    font-size: 0.65rem;
}

.ai-collapsible {
    margin-bottom: 0.375rem;
}

.ai-collapsible__toggle {
    all: unset;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.2rem;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--ks-color-text-secondary, #6b7280);
}

.ai-collapsible__icon {
    font-size: 0.6rem;
    color: var(--ks-color-text-secondary, #6b7280);
}

.ai-pre {
    margin: 0.25rem 0 0;
    padding: 0.35rem 0.5rem;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.65rem;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

.ai-pre--muted {
    background: var(--ks-color-surface-subtle, #f3f4f6);
    color: var(--ks-color-text-secondary, #6b7280);
}

.ai-pre--json {
    background: var(--ks-color-surface-code, #1e1e2e);
    color: var(--ks-color-text-code, #cdd6f4);
}

.ai-response-text {
    margin: 0.1rem 0 0;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 6rem;
    overflow-y: auto;
}

.ai-token-row {
    display: flex;
    gap: 0.6rem;
}

.ai-badge {
    display: inline-block;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.65rem;
    font-weight: 600;
}

.ai-badge--stop {
    background: #d1fae5;
    color: #065f46;
}

.ai-badge--length {
    background: #fef3c7;
    color: #92400e;
}

.ai-badge--content-filter {
    background: #fee2e2;
    color: #991b1b;
}

.ai-badge--tool {
    background: #dbeafe;
    color: #1e40af;
}

.ai-badge--default {
    background: var(--ks-color-surface-subtle, #f3f4f6);
    color: var(--ks-color-text-secondary, #6b7280);
}

.ai-guardrail {
    display: flex;
    align-items: flex-start;
    gap: 0.35rem;
    padding: 0.4rem 0.5rem;
    border-radius: 4px;
    background: #fee2e2;
    color: #991b1b;
    margin-bottom: 0.375rem;
}

.ai-guardrail__icon {
    font-size: 0.8rem;
    flex-shrink: 0;
    margin-top: 0.05rem;
}

.ai-guardrail__title {
    font-weight: 600;
}

.ai-guardrail__msg {
    font-size: 0.65rem;
    margin-top: 0.1rem;
    opacity: 0.85;
}

.ai-tool-call {
    border: 1px solid var(--ks-color-border, #e5e7eb);
    border-radius: 4px;
    margin-bottom: 0.3rem;
}

.ai-tool-call__header {
    all: unset;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: 0.3rem 0.5rem;
    box-sizing: border-box;
}

.ai-tool-call__name {
    font-weight: 600;
    color: var(--ks-color-text-primary, #111827);
}

.ai-tool-call__body {
    padding: 0 0.5rem 0.4rem;
    border-top: 1px solid var(--ks-color-border, #e5e7eb);
}

.ai-tool-call__label {
    font-weight: 500;
    color: var(--ks-color-text-secondary, #6b7280);
    margin-top: 0.3rem;
    margin-bottom: 0.1rem;
}

.ai-intermediate {
    display: flex;
    gap: 0.4rem;
    align-items: flex-start;
    margin-bottom: 0.25rem;
}

.ai-intermediate__step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.1rem;
    height: 1.1rem;
    border-radius: 50%;
    background: var(--ks-color-primary, #6366f1);
    color: #fff;
    font-size: 0.6rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 0.1rem;
}

.ai-intermediate__text {
    white-space: pre-wrap;
    word-break: break-word;
}

.ai-source {
    border-left: 2px solid var(--ks-color-primary, #6366f1);
    padding: 0.25rem 0.5rem;
    margin-bottom: 0.35rem;
}

.ai-source__content {
    margin: 0 0 0.2rem;
    font-style: italic;
    color: var(--ks-color-text-primary, #111827);
}

.ai-source__meta {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.1rem 0.4rem;
    margin: 0;
    color: var(--ks-color-text-secondary, #6b7280);
}

.ai-source__meta dt {
    font-weight: 500;
}

.ai-source__meta dd {
    margin: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.ai-show-more {
    all: unset;
    cursor: pointer;
    font-size: 0.65rem;
    color: var(--ks-color-primary, #6366f1);
    text-decoration: underline;
    margin-top: 0.2rem;
}
</style>
