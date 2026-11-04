<script setup lang="ts">
import type { KnownSlotProps } from "@kestra-io/artifact-sdk";
import AITopologyDetails from "./AITopologyDetails.vue";

// Serves both `topology-task-modal` and `execution-artifact/details`. The artifact
// props are the topology-details props minus `progress`, so it is made optional here.
defineOptions({ inheritAttrs: false });
const props = defineProps<
    Partial<Pick<KnownSlotProps["topology-details"], "progress">> &
        KnownSlotProps["execution-artifact"]
>();
</script>

<template>
    <AITopologyDetails v-bind="props" :progress="props.progress ?? []" displayMode="full" />
</template>
