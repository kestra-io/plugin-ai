<script setup lang="ts">
import type { KnownSlotProps } from "@kestra-io/artifact-sdk";
import AITopologyDetails from "./AITopologyDetails.vue";

// Serves `execution-artifact/details`. Its props are the topology-details props minus
// `progress`, which AITopologyDetails still declares, so default it here.
defineOptions({ inheritAttrs: false });
const props = defineProps<
    Partial<Pick<KnownSlotProps["topology-details"], "progress">> &
        KnownSlotProps["execution-artifact"]
>();
</script>

<template>
    <AITopologyDetails v-bind="props" :progress="props.progress ?? []" />
</template>