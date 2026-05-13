import defaultViteConfig from "@kestra-io/artifact-sdk/vite.config";

export default defaultViteConfig({
    plugin: "io.kestra.plugin.ai",

    exposes: {
        "agent.AIAgent": [
            {
                slotName: "topology-details",
                path: "./src/components/AITopologyDetails.vue",
                additionalProperties: {
                    // Height without execution: header (44) + Provider + Model rows (~48)
                    height: 92,
                    // Height with execution: adds token usage + finish reason + response preview
                    heightWithExecution: 220,
                },
            },
        ],
        "completion.ChatCompletion": [
            {
                slotName: "topology-details",
                path: "./src/components/AITopologyDetails.vue",
                additionalProperties: {
                    height: 92,
                    heightWithExecution: 180,
                },
            },
        ],
        "rag.ChatCompletion": [
            {
                slotName: "topology-details",
                path: "./src/components/AITopologyDetails.vue",
                additionalProperties: {
                    height: 108,
                    heightWithExecution: 220,
                },
            },
        ],
    },
});
