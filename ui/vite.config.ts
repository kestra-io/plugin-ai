import defaultViteConfig from "@kestra-io/artifact-sdk/vite.config";

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

export default defaultViteConfig({ plugin: "io.kestra.plugin.ai", distribute: "OSS", exposes });