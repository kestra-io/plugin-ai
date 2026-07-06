import { describe, test } from "vitest";
import { composeStories } from "@storybook/vue3";
import * as topologyStories from "../../src/AITopologyDetails.stories";

const storyModules = [
    { name: "AITopologyDetails", stories: composeStories(topologyStories) },
];

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
