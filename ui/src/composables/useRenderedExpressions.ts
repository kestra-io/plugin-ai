import { ref, watch } from "vue";
import { renderExpressions } from "@kestra-io/kestra-sdk/expressions";

/** Only values that actually contain a Pebble expression are worth a round-trip. */
const EXPRESSION_RE = /\{\{.*?}}/;

export interface RenderContext {
    executionId?: string;
    namespace?: string;
    flowId?: string;
    /** Live (possibly unsaved) flow source — resolves draft edits before they are saved. */
    flow?: string;
}

/**
 * Tenant the host UI resolved and persisted on navigation (EE writes it under `selectedTenant`).
 * We pass it explicitly because the plugin bundles its own SDK copy whose global tenant stays at the
 * `"main"` default. Absent on single-tenant OSS — there the `"main"` default is already correct.
 */
function currentTenant(): string | undefined {
    if (typeof window === "undefined") return undefined;
    return window.localStorage.getItem("selectedTenant") ?? undefined;
}

/**
 * Resolves Pebble expressions for display by calling the backend `POST /expressions/render`
 * endpoint through the framework-agnostic SDK. All rendering happens server-side; this composable
 * only wires the call into Vue reactivity and falls back to the raw value.
 *
 * Resolution is all-or-nothing per expression: an expression referencing anything the restricted
 * display engine cannot resolve (env(), kv(), missing vars, …) is returned unchanged. secret() is
 * masked as `[secret: KEY]`. Any failure keeps the raw value (no error surfaced).
 *
 * Context priority (server-side): executionId → flow source → namespace + flowId → globals only.
 */
export function useRenderedExpressions(
    expressions: () => Array<string | undefined>,
    context: () => RenderContext,
) {
    const rendered = ref<Record<string, string>>({});
    // Guards against out-of-order responses: rapid context changes (task / execution switches) fire
    // overlapping load() calls, so only the latest request is allowed to mutate `rendered`.
    let requestId = 0;

    async function load() {
        const values = (expressions() ?? []).filter(
            (v): v is string => typeof v === "string" && EXPRESSION_RE.test(v),
        );
        if (values.length === 0) {
            rendered.value = {};
            return;
        }
        const id = ++requestId;
        try {
            const { rendered: result } = await renderExpressions(
                {
                    expressions: values,
                    tenant: currentTenant(),
                    ...context(),
                },
                {
                    // Best-effort display call: never let a failed render surface the host's global
                    // error UI. Treating 404 as a non-error skips the SDK's full-page 404 overlay,
                    // and showMessageOnError suppresses the error toast for other statuses (403/5xx).
                    validateStatus: (s: number) => s === 200 || s === 404,
                    showMessageOnError: false,
                },
            );
            if (id === requestId) rendered.value = result ?? {};
        } catch {
            // Best-effort: drop any stale resolved values so display() falls back to the raw
            // template instead of showing values resolved under a previous context.
            if (id === requestId) rendered.value = {};
        }
    }

    watch([() => (expressions() ?? []).join(" "), () => JSON.stringify(context() ?? {})], load, {
        immediate: true,
    });

    function display(value?: string): string | undefined {
        if (value === undefined) return undefined;
        return rendered.value[value] ?? value;
    }

    return { display };
}