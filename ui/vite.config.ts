import defaultViteConfig from "@kestra-io/artifact-sdk/vite.config";

const TOPOLOGY_DETAILS = "./src/components/AITopologyDetails.vue";

export default defaultViteConfig({
    plugin: "io.kestra.plugin.ai",

    exposes: {
        "agent.AIAgent": [
            {
                slotName: "topology-details",
                path: TOPOLOGY_DETAILS,
                additionalProperties: {
                    height: 92,
                    heightWithExecution: 220,
                },
            },
            {
                slotName: "topology-task-drawer",
                path: TOPOLOGY_DETAILS,
            },
        ],
        "completion.ChatCompletion": [
            {
                slotName: "topology-details",
                path: TOPOLOGY_DETAILS,
                additionalProperties: {
                    height: 92,
                    heightWithExecution: 180,
                },
            },
            {
                slotName: "topology-task-drawer",
                path: TOPOLOGY_DETAILS,
            },
        ],
        "completion.Classification": [
            {
                slotName: "topology-details",
                path: TOPOLOGY_DETAILS,
                additionalProperties: {
                    height: 92,
                    heightWithExecution: 180,
                },
            },
            {
                slotName: "topology-task-drawer",
                path: TOPOLOGY_DETAILS,
            },
        ],
        "completion.ImageGeneration": [
            {
                slotName: "topology-details",
                path: TOPOLOGY_DETAILS,
                additionalProperties: {
                    height: 92,
                    heightWithExecution: 180,
                },
            },
            {
                slotName: "topology-task-drawer",
                path: TOPOLOGY_DETAILS,
            },
        ],
        "completion.JSONStructuredExtraction": [
            {
                slotName: "topology-details",
                path: TOPOLOGY_DETAILS,
                additionalProperties: {
                    height: 92,
                    heightWithExecution: 220,
                },
            },
            {
                slotName: "topology-task-drawer",
                path: TOPOLOGY_DETAILS,
            },
        ],
        "rag.ChatCompletion": [
            {
                slotName: "topology-details",
                path: TOPOLOGY_DETAILS,
                additionalProperties: {
                    height: 108,
                    heightWithExecution: 220,
                },
            },
            {
                slotName: "topology-task-drawer",
                path: TOPOLOGY_DETAILS,
            },
        ],
        "rag.IngestDocument": [
            {
                slotName: "topology-details",
                path: TOPOLOGY_DETAILS,
                additionalProperties: {
                    height: 92,
                },
            },
            {
                slotName: "topology-task-drawer",
                path: TOPOLOGY_DETAILS,
            },
        ],
        "rag.Search": [
            {
                slotName: "topology-details",
                path: TOPOLOGY_DETAILS,
                additionalProperties: {
                    height: 92,
                },
            },
            {
                slotName: "topology-task-drawer",
                path: TOPOLOGY_DETAILS,
            },
        ],
    },
});
