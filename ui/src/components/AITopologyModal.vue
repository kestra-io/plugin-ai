<script setup lang="ts">
import type { KnownSlotProps } from "@kestra-io/artifact-sdk";
import AITopologyDetails from "./AITopologyDetails.vue";

// `source` (the current flow YAML) is passed through by the host but not part of the shared
// KnownSlotProps type, so it is declared explicitly to forward it to AITopologyDetails.
defineProps<KnownSlotProps["topology-details"] & { source?: string }>();
</script>

<template>
    <AITopologyDetails v-bind="$props" displayMode="full" />
</template>
