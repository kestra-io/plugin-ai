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
 * The Kestra UI encodes the active tenant as the first path segment after the app base
 * (`/ui/<tenant>/flows/...`). The generated SDK otherwise resolves the tenant to its built-in
 * `"main"` default because the host configures the shared client via `useClient()`/manual URLs
 * and never calls `setSelectedTenant()`. On EE instances whose tenant is named differently that
 * default 404s, so we resolve the tenant from the URL and pass it explicitly per call.
 */
function currentTenant(): string | undefined {
    if (typeof window === "undefined") return undefined;
    // KESTRA_UI_PATH is the absolute UI base (e.g. "/ui/"). It can be a relative "./" in some
    // templated builds, so only trust it when absolute and otherwise assume the conventional mount.
    const raw = (window as { KESTRA_UI_PATH?: string }).KESTRA_UI_PATH;
    const base = raw && raw.startsWith("/") ? raw.replace(/\/$/, "") : "/ui";
    let path = window.location.pathname;
    if (base && path.startsWith(base)) {
        path = path.slice(base.length);
    }
    return path.split("/").find(Boolean) || undefined;
}

/**
 * Kestra's `CsrfTokenFilter` rejects cookie-authenticated non-safe requests (POST) that lack a
 * CSRF token with a 403. `expressions/render` is a POST, so the browser (JWT/basic-auth cookie)
 * must send the token. The host app reads it from `<meta name="csrf-token">` and forwards it as
 * the `X-CSRF-TOKEN` header, but the generated SDK client we call has no such interceptor, so we
 * attach it ourselves. Returns an empty object when there is no token (e.g. Authorization-header
 * clients, which the filter exempts).
 */
function csrfHeaders(): Record<string, string> {
    if (typeof document === "undefined") return {};
    const token = document.querySelector('meta[name="csrf-token"]')?.getAttribute("content");
    return token ? { "X-CSRF-TOKEN": token } : {};
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
                    headers: csrfHeaders(),
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

    /** Returns the rendered value for `value`, falling back to the raw value. */
    function display(value?: string): string | undefined {
        if (value === undefined) return undefined;
        return rendered.value[value] ?? value;
    }

    return { display };
}
