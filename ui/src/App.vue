<script setup lang="ts">
import AITopologyDetails from "./components/AITopologyDetails.vue";
import { SLOTS } from "@kestra-io/artifact-sdk";
</script>

<template>
    <div style="padding: 1rem; font-family: sans-serif">
        <AITopologyDetails v-bind="SLOTS['topology-details'].defaultProps" />
    </div>
</template>
