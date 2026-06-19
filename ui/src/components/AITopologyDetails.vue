<script setup lang="ts">
import type { KnownSlotProps } from "@kestra-io/artifact-sdk";
import { KsTopologyDetails, KsTag, KsCollapse, KsCollapseItem, KsAlert } from "@kestra-io/design-system";
import { computed, ref, watch, useAttrs } from "vue";
import { execution as fetchExecution } from "@kestra-io/kestra-sdk";

const props = defineProps<KnownSlotProps["topology-details"]>();
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

const summaryRows = computed(() => {
    const rows: { label: string; value: string }[] = [
        { label: "Provider", value: provider.value ?? "—" },
        { label: "Model", value: modelName.value ?? "—" },
    ];
    if (isRag.value && retrieverType.value) {
        rows.push({ label: "Retriever", value: retrieverType.value });
    }
    return rows;
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
    } catch (e) {
        console.error("[AITopologyDetails] failed to load task outputs", e);
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

const thinkingTruncated = ref(true);
const THINKING_TRUNCATE_LENGTH = 300;

const thinkingDisplay = computed(() => {
    if (!thinking.value) return undefined;
    if (!thinkingTruncated.value || thinking.value.length <= THINKING_TRUNCATE_LENGTH) {
        return thinking.value;
    }
    return thinking.value.slice(0, THINKING_TRUNCATE_LENGTH) + "…";
});

function formatJson(v: unknown): string {
    try {
        return JSON.stringify(v, null, 2);
    } catch {
        return String(v);
    }
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

const hasResponse = computed(
    () => !guardrailViolated.value && (!!textOutput.value || !!jsonOutput.value),
);
</script>

<template>
    <div class="ai-details">
        <!-- Pre-execution: provider + model always shown -->
        <KsTopologyDetails :rows="summaryRows" />

        <!-- System prompt collapsible -->
        <KsCollapse v-if="systemPrompt">
            <KsCollapseItem name="system-prompt" title="System prompt">
                <pre class="ai-pre">{{ systemPrompt }}</pre>
            </KsCollapseItem>
        </KsCollapse>

        <!-- Tools list -->
        <div v-if="toolNames.length > 0" class="ai-tools">
            <KsTag v-for="name in toolNames" :key="name" size="small" type="info">
                {{ name }}
            </KsTag>
        </div>

        <!-- Post-execution panel -->
        <template v-if="hasExecution && outputs">
            <!-- Guardrail violation -->
            <KsAlert
                v-if="guardrailViolated"
                type="error"
                :title="guardrailMessage ?? 'Guardrail violated'"
                show-icon
            />

            <!-- LLM response -->
            <template v-if="hasResponse">
                <p v-if="textOutput" class="ai-response-text">{{ textOutput }}</p>
                <pre v-else-if="jsonOutput" class="ai-pre ai-pre--json">{{ formatJson(jsonOutput) }}</pre>
            </template>

            <!-- Finish reason + token usage -->
            <div v-if="finishReason || tokenUsage" class="ai-meta-row">
                <KsTag v-if="finishReason" :type="finishReasonType" size="small">
                    {{ finishReason }}
                </KsTag>
                <span v-if="tokenUsage" class="ai-tokens">
                    <span>in: {{ tokenUsage.inputTokenCount ?? "—" }}</span>
                    <span>out: {{ tokenUsage.outputTokenCount ?? "—" }}</span>
                    <span>total: {{ tokenUsage.totalTokenCount ?? "—" }}</span>
                </span>
            </div>

            <!-- Full-view extras -->
            <template v-if="isFullView">
                <!-- Tool call timeline -->
                <KsCollapse v-if="toolExecutions.length > 0">
                    <KsCollapseItem
                        v-for="(exec, i) in toolExecutions"
                        :key="exec.requestId ?? i"
                        :name="String(i)"
                        :title="exec.requestName"
                    >
                        <pre class="ai-pre ai-pre--json">{{ formatJson(exec.requestArguments) }}</pre>
                        <pre v-if="exec.result" class="ai-pre">{{ exec.result }}</pre>
                    </KsCollapseItem>
                </KsCollapse>

                <!-- Thinking / reasoning chain -->
                <KsCollapse v-if="thinking || intermediateResponses.length > 0">
                    <KsCollapseItem name="thinking" title="Reasoning Chain">
                        <template v-if="thinking">
                            <pre class="ai-pre">{{ thinkingDisplay }}</pre>
                            <button
                                v-if="thinking.length > THINKING_TRUNCATE_LENGTH"
                                class="ai-show-more"
                                @click="thinkingTruncated = !thinkingTruncated"
                            >
                                {{ thinkingTruncated ? "Show more" : "Show less" }}
                            </button>
                        </template>
                        <div
                            v-for="(resp, i) in intermediateResponses"
                            :key="resp.id ?? i"
                            class="ai-intermediate"
                        >
                            <span class="ai-intermediate__step">{{ i + 1 }}</span>
                            <span class="ai-intermediate__text">{{ resp.completion }}</span>
                        </div>
                    </KsCollapseItem>
                </KsCollapse>

                <!-- RAG sources -->
                <KsCollapse v-if="sources.length > 0">
                    <KsCollapseItem name="sources" :title="`RAG Sources (${sources.length})`">
                        <div v-for="(src, i) in sources" :key="i" class="ai-source">
                            <p class="ai-source__content">{{ src.content }}</p>
                            <dl v-if="src.metadata" class="ai-source__meta">
                                <template v-for="(val, key) in src.metadata" :key="key">
                                    <dt>{{ key }}</dt>
                                    <dd>{{ val }}</dd>
                                </template>
                            </dl>
                        </div>
                    </KsCollapseItem>
                </KsCollapse>
            </template>
        </template>
    </div>
</template>

<style scoped>
.ai-details {
    --ai-font-sm: 0.65rem;
    --ai-gap-xs: 0.25rem;
    --ai-gap-sm: 0.5rem;
    --ai-fw-medium: 500;
    --ai-fw-bold: 700;

    position: relative;
    z-index: 1;
    padding: 0.5rem 0.75rem;
    font-size: 0.7rem;
    line-height: 1.4;
}

.ai-tools {
    display: flex;
    flex-wrap: wrap;
    gap: var(--ai-gap-xs);
    margin-top: var(--ai-gap-xs);
    margin-bottom: var(--ai-gap-xs);
}

.ai-pre {
    margin: var(--ai-gap-xs) 0 0;
    padding: 0.35rem var(--ai-gap-sm);
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: var(--ai-font-sm);
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--ks-color-surface-subtle, #f3f4f6);
    color: var(--ks-color-text-secondary, #6b7280);
}

.ai-pre--json {
    background: var(--ks-color-surface-code, #1e1e2e);
    color: var(--ks-color-text-code, #cdd6f4);
}

.ai-response-text {
    margin: var(--ai-gap-xs) 0 0;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 6rem;
    overflow-y: auto;
}

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
    font-size: var(--ai-font-sm);
    color: var(--ks-color-text-secondary, #6b7280);
}

.ai-intermediate {
    display: flex;
    gap: 0.4rem;
    align-items: flex-start;
    margin-bottom: var(--ai-gap-xs);
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
    font-size: var(--ai-font-sm);
    font-weight: var(--ai-fw-bold);
    flex-shrink: 0;
}

.ai-intermediate__text {
    white-space: pre-wrap;
    word-break: break-word;
}

.ai-source {
    border-left: 2px solid var(--ks-color-primary, #6366f1);
    padding: var(--ai-gap-xs) var(--ai-gap-sm);
    margin-bottom: 0.35rem;
    font-size: var(--ai-font-sm);
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
    color: var(--ks-color-text-secondary, #6b7280);
}

.ai-source__meta dt {
    font-weight: var(--ai-fw-medium);
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
    font-size: var(--ai-font-sm);
    color: var(--ks-color-primary, #6366f1);
    text-decoration: underline;
    margin-top: 0.2rem;
}
</style>
