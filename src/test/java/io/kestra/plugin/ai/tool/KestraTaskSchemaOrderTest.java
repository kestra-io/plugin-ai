package io.kestra.plugin.ai.tool;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class KestraTaskSchemaOrderTest {
    @Test
    @SuppressWarnings("unchecked")
    void removeAlreadySet_keepsTheOrderOfTheRemainingProperties() {
        var properties = new LinkedHashMap<String, Object>();
        for (var name : List.of("eventSummary", "sentiment", "fixed", "zeta", "alpha", "company")) {
            properties.put(name, Map.of("type", "string"));
        }
        var schema = new HashMap<String, Object>(Map.of("properties", properties, "required", List.of("eventSummary")));
        // "..." means the model must fill the value; "fixed" is already set, so it is not exposed
        var taskProperties = Map.<String, Object>of(
            "eventSummary", "...", "sentiment", "...", "fixed", "set by the flow", "zeta", "...", "alpha", "...", "company", "..."
        );
        var expected = List.of("eventSummary", "sentiment", "zeta", "alpha", "company");

        // Guard: a HashMap would iterate these keys in a different order, so the test cannot pass by chance
        var hashed = new HashMap<String, Object>();
        expected.forEach(name -> hashed.put(name, name));
        assertThat(hashed.keySet()).doesNotContainSequence(expected);

        new KestraTask().removeAlreadySet(schema, taskProperties);

        assertThat(((Map<String, Object>) schema.get("properties")).keySet()).containsExactlyElementsOf(expected);
    }
}
