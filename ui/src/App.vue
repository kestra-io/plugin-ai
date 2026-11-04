<script setup lang="ts">
import AITopologyModal from "./components/AITopologyModal.vue";
import { SLOTS } from "@kestra-io/artifact-sdk";
</script>

<template>
    <div style="padding: 1rem; font-family: sans-serif">
        <AITopologyModal v-bind="SLOTS['execution-artifact'].defaultProps" />
    </div>
</template>