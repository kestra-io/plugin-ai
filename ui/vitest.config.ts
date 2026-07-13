import { defineConfig } from "vitest/config";
import vue from "@vitejs/plugin-vue";

// Runs Storybook story `play` functions headlessly (jsdom) so CI catches UI regressions
// without a browser runner. Deliberately does NOT extend the federation vite.config;
// vitest prefers this file over vite.config.ts.
export default defineConfig({
    plugins: [vue()],
    test: {
        environment: "jsdom",
        include: ["tests/**/*.{test,spec}.ts"],
        setupFiles: ["tests/unit/setup.ts"],
        // The design-system dist imports .css; inline it so Vite transforms those imports
        // instead of Node trying to require raw .css files.
        server: { deps: { inline: [/@kestra-io\/design-system/] } },
    },
});
