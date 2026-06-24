<script setup lang="ts">
import type { KnownSlotProps } from "@kestra-io/artifact-sdk";
import AITopologyDetails from "./AITopologyDetails.vue";

defineProps<KnownSlotProps["topology-details"]>();
</script>

<template>
    <AITopologyDetails v-bind="$props" displayMode="full" />
</template>
