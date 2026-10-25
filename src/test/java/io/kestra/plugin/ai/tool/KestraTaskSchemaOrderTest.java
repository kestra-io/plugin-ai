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
        // "..." means the model fills the value, "fixed" is set by the flow so it is not exposed
        var taskProperties = Map.<String, Object>of(
            "eventSummary", "...", "sentiment", "...", "fixed", "set by the flow", "zeta", "...", "alpha", "...", "company", "..."
        );
        var expected = List.of("eventSummary", "sentiment", "zeta", "alpha", "company");

        new KestraTask().removeAlreadySet(schema, taskProperties);

        assertThat(((Map<String, Object>) schema.get("properties")).keySet()).containsExactlyElementsOf(expected);
    }
}
