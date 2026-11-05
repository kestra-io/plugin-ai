// Custom vite config instead of the SDK `defaultViteConfig`: the SDK names the federation
// remote from package.json (`io.kestra.plugin.ai`), this config keeps `io_kestra_plugin_ai`.
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { federation } from "@module-federation/vite";
import { createHash } from "node:crypto";
import { createRequire } from "module";

const SLOT_NAMES = [
    "topology-details",
    "topology-task-drawer",
    "topology-task-modal",
    "execution-artifact/details",
];

const ARTIFACT_FILENAME = "plugin-ui.js";
const PLUGIN_ID = "io.kestra.plugin.ai";

const SHOW_AI_DETAILS = { label: "Show Details", taskProp: "", lang: "" };
const TOPOLOGY_DETAILS = "./src/components/AITopologyDetails.vue";
const TOPOLOGY_MODAL = "./src/components/AITopologyModal.vue";

// Kestra 2.0.x renders the topology slots, 2.1+ ignores them and renders
// `execution-artifact/details` (Artifacts button and tab) instead.
const exposes = {
    "agent.AIAgent": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                // base (56) + 5 rows x 31.5, rounded up. Compact card rows (summaryRows): Provider, Model, Tools, Memory, Retrievers.
                // heightWithExecution adds the ~30px badge row the host draws under the node with an execution.
                height: 214,
                heightWithExecution: 244,
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
        {
            slotName: "execution-artifact/details",
            path: TOPOLOGY_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "completion.ChatCompletion": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                // base (56) + 3 rows x 31.5, rounded up. Compact card rows (summaryRows): Provider, Model, Tools.
                // heightWithExecution adds the ~30px badge row the host draws under the node with an execution.
                height: 151,
                heightWithExecution: 181,
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
        {
            slotName: "execution-artifact/details",
            path: TOPOLOGY_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "completion.Classification": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                // base (56) + 2 rows x 31.5, rounded up. Compact card rows (summaryRows): Provider, Model.
                // heightWithExecution adds the ~30px badge row the host draws under the node with an execution.
                height: 120,
                heightWithExecution: 150,
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
        {
            slotName: "execution-artifact/details",
            path: TOPOLOGY_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "completion.ImageGeneration": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                // base (56) + 2 rows x 31.5, rounded up. Compact card rows (summaryRows): Provider, Model.
                // heightWithExecution adds the ~30px badge row the host draws under the node with an execution.
                height: 120,
                heightWithExecution: 150,
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
        {
            slotName: "execution-artifact/details",
            path: TOPOLOGY_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "completion.JSONStructuredExtraction": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                // base (56) + 2 rows x 31.5, rounded up. Compact card rows (summaryRows): Provider, Model.
                // heightWithExecution adds the ~30px badge row the host draws under the node with an execution.
                height: 120,
                heightWithExecution: 150,
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
        {
            slotName: "execution-artifact/details",
            path: TOPOLOGY_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "rag.ChatCompletion": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                // base (56) + 5 rows x 31.5, rounded up. Compact card rows (summaryRows): Provider, Model, Tools, Memory, Retrievers.
                // heightWithExecution adds the ~30px badge row the host draws under the node with an execution.
                height: 214,
                heightWithExecution: 244,
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
        {
            slotName: "execution-artifact/details",
            path: TOPOLOGY_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "rag.IngestDocument": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                // base (56) + 2 rows x 31.5, rounded up. Compact card rows (summaryRows): Provider, Model.
                // heightWithExecution adds the ~30px badge row the host draws under the node with an execution.
                height: 120,
                heightWithExecution: 150,
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
        {
            slotName: "execution-artifact/details",
            path: TOPOLOGY_MODAL,
            additionalProperties: { title: "Execution details" },
        },
    ],
    "rag.Search": [
        {
            slotName: "topology-details",
            path: TOPOLOGY_DETAILS,
            additionalProperties: {
                // base (56) + 2 rows x 31.5, rounded up. Compact card rows (summaryRows): Provider, Model.
                // heightWithExecution adds the ~30px badge row the host draws under the node with an execution.
                height: 120,
                heightWithExecution: 150,
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
        {
            slotName: "execution-artifact/details",
            path: TOPOLOGY_MODAL,
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
        ...(isStorybook
            ? []
            : [
                  manifestPlugin(),
                  federation({
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
                  }),
              ]),
        vue(),
    ],
});