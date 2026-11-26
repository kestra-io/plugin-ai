import { afterEach, describe, expect, test } from "vitest";
import { createApp, nextTick, type App } from "vue";
import AITopologyDetails from "../../src/components/AITopologyDetails.vue";
import AITopologyModal from "../../src/components/AITopologyModal.vue";

let app: App | undefined;
let container: HTMLDivElement;

afterEach(() => {
    app?.unmount();
    container?.remove();
});

async function render(outputs: Record<string, unknown>, options: { full?: boolean; modal?: boolean; fetched?: boolean } = {}) {
    container = document.createElement("div");
    document.body.append(container);
    app = createApp(options.modal ? AITopologyModal : AITopologyDetails, {
        task: { id: "task", type: "io.kestra.plugin.ai.completion.ImageGeneration" },
        displayMode: options.full === false ? undefined : "full",
        progress: [],
        execution: { id: "exec", taskRunList: [{ id: "run", taskId: "task", outputs: options.fetched ? undefined : outputs }] },
        ...(options.fetched ? { fetchOutputs: async () => outputs } : {}),
    });
    app.mount(container);
    await nextTick();
    await nextTick();
    return container;
}

describe("task results", () => {
    test.each(["javascript:alert(1)", "data:image/png;base64,AAAA", "//example.com/image.png", "/image.png", "https://", "ftp://example.com/image.png"])("does not render image URL %s", async imageUrl => {
        const view = await render({ imageUrl });
        expect(view.querySelector("img")).toBeNull();
    });

    test.each(["http://example.com/image.png", "https://example.com/image.png"])("renders explicit URL %s", async imageUrl => {
        const view = await render({ imageUrl });
        expect(view.querySelector("img")?.getAttribute("src")).toBe(imageUrl);
        expect(view.querySelector("img")?.getAttribute("referrerpolicy")).toBe("no-referrer");
    });

    test.each(["false", "0", "null"])("formats valid extracted JSON %s", async extractedJson => {
        const view = await render({ extractedJson });
        expect(view.querySelector(".ai-pre--json")?.textContent).toBe(extractedJson);
    });

    test.each(['{"broken":', '```json\n{"name": "Ada"}\n```', '{\n  "name": "Ada'])
        ("preserves malformed extracted JSON as raw text: %s", async extractedJson => {
            const view = await render({ extractedJson });
            expect(view.querySelector(".ai-pre--json")?.textContent).toBe(extractedJson);
        });

    test("hides empty text output", async () => {
        const view = await render({ textOutput: "" });
        expect(view.querySelector(".ai-section")).toBeNull();
    });

    test("shows JSON when text output is empty", async () => {
        const view = await render({ textOutput: "", jsonOutput: { name: "Ada" } });
        expect(view.querySelector(".ai-pre--json")?.textContent).toBe(JSON.stringify({ name: "Ada" }, null, 2));
    });

    test("shows zero documents and token counts", async () => {
        const view = await render({ ingestedDocuments: 0, inputTokenCount: 0, outputTokenCount: 0, totalTokenCount: 0 });
        expect(view.textContent).toContain("Ingested documents");
        expect(view.querySelector(".ai-kv__val")?.textContent).toBe("0");
        expect(view.querySelector(".ai-token-box__value")?.textContent).toBe("0");
    });

    test("shows an empty search result", async () => {
        const view = await render({ results: [], size: 0 });
        expect(view.querySelector(".ai-kv__val")?.textContent).toBe("0");
        expect(view.querySelector(".ai-pre--json")?.textContent).toBe("[]");
    });

    test("shows search size without inline results", async () => {
        const view = await render({ size: 4, uri: "kestra:///results.ion" });
        expect(view.querySelector(".ai-kv__val")?.textContent).toBe("4");
        expect(view.querySelector(".ai-pre--json")).toBeNull();
    });

    test("keeps absent outputs empty", async () => {
        const view = await render({});
        expect(view.querySelector(".ai-section")).toBeNull();
    });

    test("does not expose a response after a guardrail violation", async () => {
        const view = await render({ classification: "private", guardrailViolated: true });
        expect(view.textContent).not.toContain("private");
    });

    test("keeps results out of the compact card", async () => {
        const view = await render({ classification: "positive" }, { full: false });
        expect(view.textContent).not.toContain("positive");
    });

    test("renders fetched results through the modal and execution-artifact wrapper", async () => {
        const view = await render({ classification: "positive" }, { modal: true, fetched: true });
        expect(view.textContent).toContain("positive");
    });
});
