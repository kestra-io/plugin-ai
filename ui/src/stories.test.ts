import { describe, test } from "vitest";
import { composeStories } from "@storybook/vue3";
import * as topologyStories from "./AITopologyDetails.stories";

// Run every story that defines a `play` function through the same assertions Storybook
// uses interactively. `.run()` renders the story into the DOM and executes `play`, throwing
// (failing the test) on any assertion failure. New stories with a `play` are picked up
// automatically. Stories without a `play` are not exercised here.
const storyModules = [{ name: "AITopologyDetails", stories: composeStories(topologyStories) }];

for (const { name, stories } of storyModules) {
    describe(name, () => {
        for (const [storyName, Story] of Object.entries(stories)) {
            if (typeof Story.play !== "function") continue;
            test(storyName, async () => {
                await Story.run();
            });
        }
    });
}
