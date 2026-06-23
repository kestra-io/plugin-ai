// Custom vite config that extends @kestra-io/artifact-sdk/vite.config while
// adding support for the `topology-task-modal` slot introduced in
// https://github.com/kestra-io/kestra/pull/16795.
//
// Once @kestra-io/artifact-sdk ships a version that includes this slot name in
// its KnownSlotNames list, replace this file with the one-liner:
//   import defaultViteConfig from "@kestra-io/artifact-sdk/vite.config";
//   export default defaultViteConfig({ plugin: "io.kestra.plugin.ai", exposes: { ... } });
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { federation } from "@module-federation/vite";
import * as fs from "node:fs";
import * as path from "node:path";
import { createHash } from "node:crypto";
import { createRequire } from "module";

const SLOT_NAMES = [
    "topology-details",
    "topology-task-drawer",
    "topology-task-modal",
];

const ARTIFACT_FILENAME = "plugin-ui.js";
const PLUGIN_ID = "io.kestra.plugin.ai";

const SHOW_AI_DETAILS = { label: "Show Details", taskProp: "", lang: "" };
const TOPOLOGY_DETAILS = "./src/components/AITopologyDetails.vue";
const TOPOLOGY_MODAL = "./src/components/AITopologyModal.vue";

const exposes = {
    "agent.AIAgent": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                // base (56) + 2 rows × ~32px
                height: 120,
                heightWithExecution: 120,
                customAction: SHOW_AI_DETAILS,
            },
        },
        {
            slotName: "topology-task-drawer",
            path: TOPOLOGY_DETAILS,
        },
        {
            slotName: "topology-task-modal",
            path: TOPOLOGY_MODAL,
        },
    ],
    "completion.ChatCompletion": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                height: 120,
                heightWithExecution: 120,
                customAction: SHOW_AI_DETAILS,
            },
        },
        {
            slotName: "topology-task-drawer",
            path: TOPOLOGY_DETAILS,
        },
        {
            slotName: "topology-task-modal",
            path: TOPOLOGY_MODAL,
        },
    ],
    "completion.Classification": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                height: 120,
                heightWithExecution: 120,
                customAction: SHOW_AI_DETAILS,
            },
        },
        {
            slotName: "topology-task-drawer",
            path: TOPOLOGY_DETAILS,
        },
        {
            slotName: "topology-task-modal",
            path: TOPOLOGY_MODAL,
        },
    ],
    "completion.ImageGeneration": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                height: 120,
                heightWithExecution: 120,
                customAction: SHOW_AI_DETAILS,
            },
        },
        {
            slotName: "topology-task-drawer",
            path: TOPOLOGY_DETAILS,
        },
        {
            slotName: "topology-task-modal",
            path: TOPOLOGY_MODAL,
        },
    ],
    "completion.JSONStructuredExtraction": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                height: 120,
                heightWithExecution: 120,
                customAction: SHOW_AI_DETAILS,
            },
        },
        {
            slotName: "topology-task-drawer",
            path: TOPOLOGY_DETAILS,
        },
        {
            slotName: "topology-task-modal",
            path: TOPOLOGY_MODAL,
        },
    ],
    "rag.ChatCompletion": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                // base (56) + 3 rows × ~32px (Provider + Model + Retriever)
                height: 151,
                heightWithExecution: 151,
                customAction: SHOW_AI_DETAILS,
            },
        },
        {
            slotName: "topology-task-drawer",
            path: TOPOLOGY_DETAILS,
        },
        {
            slotName: "topology-task-modal",
            path: TOPOLOGY_MODAL,
        },
    ],
    "rag.IngestDocument": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                height: 120,
                customAction: SHOW_AI_DETAILS,
            },
        },
        {
            slotName: "topology-task-drawer",
            path: TOPOLOGY_DETAILS,
        },
        {
            slotName: "topology-task-modal",
            path: TOPOLOGY_MODAL,
        },
    ],
    "rag.Search": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                height: 120,
                customAction: SHOW_AI_DETAILS,
            },
        },
        {
            slotName: "topology-task-drawer",
            path: TOPOLOGY_DETAILS,
        },
        {
            slotName: "topology-task-modal",
            path: TOPOLOGY_MODAL,
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

function getSharedKestraSdk(): Record<string, { singleton: boolean }> {
    try {
        const pkg = tryResolve("@kestra-io/kestra-sdk/package.json");
        if (!pkg) return {};
        const { exports = {} } = JSON.parse(fs.readFileSync(pkg, "utf-8"));
        return Object.keys(exports).reduce(
            (acc: Record<string, { singleton: boolean }>, key: string) => {
                const name =
                    key === "."
                        ? "@kestra-io/kestra-sdk"
                        : `@kestra-io/kestra-sdk/${key.replace("./", "")}`;
                acc[name] = { singleton: true };
                return acc;
            },
            {},
        );
    } catch {
        return {};
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
                ...getSharedKestraSdk(),
            },
            dts: false,
        })]),
        vue(),
    ],
});
