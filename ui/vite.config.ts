// Custom vite config that extends @kestra-io/artifact-sdk/vite.config while
// adding support for the `execution-artifact/details` slot.
//
// Once @kestra-io/artifact-sdk ships a version that includes this slot name in
// its KnownSlotNames list, replace this file with the one-liner:
//   import defaultViteConfig from "@kestra-io/artifact-sdk/vite.config";
//   export default defaultViteConfig({ plugin: "io.kestra.plugin.ai", exposes: { ... } });
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { federation } from "@module-federation/vite";
import { createHash } from "node:crypto";
import { createRequire } from "module";

const SLOT_NAMES = ["execution-artifact/details"];

const ARTIFACT_FILENAME = "plugin-ui.js";
const PLUGIN_ID = "io.kestra.plugin.ai";

const ARTIFACT_MODAL = "./src/components/AITopologyModal.vue";

// Only the execution-artifact tab is registered. The topology cards, drawer and
// modal are gone from the topology redesign (kestra-io/kestra#22001).
const exposes = {
    "agent.AIAgent": [
        {
            slotName: "execution-artifact/details",
            path: ARTIFACT_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "completion.ChatCompletion": [
        {
            slotName: "execution-artifact/details",
            path: ARTIFACT_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "completion.Classification": [
        {
            slotName: "execution-artifact/details",
            path: ARTIFACT_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "completion.ImageGeneration": [
        {
            slotName: "execution-artifact/details",
            path: ARTIFACT_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "completion.JSONStructuredExtraction": [
        {
            slotName: "execution-artifact/details",
            path: ARTIFACT_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "rag.ChatCompletion": [
        {
            slotName: "execution-artifact/details",
            path: ARTIFACT_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "rag.IngestDocument": [
        {
            slotName: "execution-artifact/details",
            path: ARTIFACT_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "rag.Search": [
        {
            slotName: "execution-artifact/details",
            path: ARTIFACT_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
};

const require = createRequire(import.meta.url);

function tryResolve(id: string): string | null {
    try {
        return require.resolve(id);
    } catch (e: any) {
        if (e.code === "MODULE_NOT_FOUND" && e.message.includes(`'${id}'`)) return null;
        throw e;
    }
}

function manifestPlugin() {
    return {
        name: "@kestra-io/manifest-plugin",
        enforce: "post" as const,
        generateBundle(_: unknown, bundle: Record<string, any>) {
            const cssFileNames = Object.keys(bundle).filter((f) => f.endsWith(".css"));
            const hash = createHash("sha256");
            for (const fileName of Object.keys(bundle).sort()) {
                const chunk = bundle[fileName];
                const content = chunk.type === "chunk" ? chunk.code : chunk.source;
                hash.update(typeof content === "string" ? content : Buffer.from(content));
            }
            const sourceHash = hash.digest("hex");
            const tasks: Record<string, any[]> = {};
            for (const task in exposes) {
                const completeTaskName = `${PLUGIN_ID}.${task}`;
                const manifestTask: any[] = tasks[completeTaskName] ?? [];
                for (const module of exposes[task as keyof typeof exposes]) {
                    if (!SLOT_NAMES.includes(module.slotName)) {
                        throw new Error(
                            `Unknown slot "${module.slotName}". Allowed: ${SLOT_NAMES.join(", ")}`,
                        );
                    }
                    manifestTask.push({
                        uiModule: module.slotName,
                        styles: cssFileNames,
                        staticInfo: (module as any).additionalProperties,
                    });
                }
                tasks[completeTaskName] = manifestTask;
            }
            (this as any).emitFile({
                type: "asset",
                fileName: "manifest.json",
                name: "manifest",
                source: JSON.stringify({ sourceHash, ...tasks }, null, 2),
            });
        },
    };
}

const isStorybook = process.env.STORYBOOK === "true";

export default defineConfig({
    build: {
        emptyOutDir: true,
        modulePreload: false,
        cssCodeSplit: false,
    },
    plugins: [
        ...(isStorybook ? [] : [manifestPlugin(), federation({
            filename: ARTIFACT_FILENAME,
            name: PLUGIN_ID.replaceAll(".", "_"),
            manifest: true,
            exposes: Object.entries(exposes).reduce(
                (acc: Record<string, string>, [task, modules]) => {
                    for (const { slotName, path: modulePath } of modules) {
                        acc[`./${task}/${slotName}`] = modulePath;
                    }
                    return acc;
                },
                {},
            ),
            shared: {
                vue: { singleton: true, requiredVersion: "^3" },
                "@kestra-io/design-system": { singleton: true },
                ...(tryResolve("vue-i18n/package.json")
                    ? { "vue-i18n": { singleton: true, requiredVersion: "^11" } }
                    : {}),
            },
            dts: false,
        })]),
        vue(),
    ],
});
