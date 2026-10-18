package io.kestra.plugin.ai.tool.internal;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import dev.langchain4j.model.chat.request.json.JsonObjectSchema;

import static org.assertj.core.api.Assertions.assertThat;

class JsonObjectSchemaTranslatorTest {
    @Test
    void keepsPropertyOrderOfTopLevelNestedAndDefinitionSchemas() {
        Map<String, Object> company = ordered(
            "type", "object",
            "properties", ordered(
                "ticker", ordered("type", "string"),
                "name", ordered("type", "string"),
                "cik", ordered("type", "integer"),
                "exchange", ordered("type", "string")
            )
        );
        Map<String, Object> address = ordered(
            "type", "object",
            "properties", ordered(
                "street", ordered("type", "string"),
                "city", ordered("type", "string"),
                "country", ordered("type", "string"),
                "zip", ordered("type", "string")
            )
        );
        Map<String, Object> schema = ordered(
            "type", "object",
            "required", List.of("event_summary", "sentiment"),
            "properties", ordered(
                "event_summary", ordered("type", "string"),
                "sentiment", ordered("type", "string", "enum", List.of("positive", "neutral", "negative")),
                "zeta", ordered("type", "string"),
                "alpha", ordered("type", "number"),
                "company", company,
                "headquarters", ordered("$ref", "#/$defs/address")
            ),
            "$defs", ordered("address", address)
        );

        // Guard: a HashMap would iterate these keys in a different order, so the test cannot pass by chance
        assertThat(new HashMap<>(properties(schema)).keySet())
            .doesNotContainSequence("event_summary", "sentiment", "zeta", "alpha", "company", "headquarters");
        assertThat(new HashMap<>(properties(company)).keySet()).doesNotContainSequence("ticker", "name", "cik", "exchange");
        assertThat(new HashMap<>(properties(address)).keySet()).doesNotContainSequence("street", "city", "country", "zip");

        JsonObjectSchema result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(result.properties().keySet())
            .containsExactly("event_summary", "sentiment", "zeta", "alpha", "company", "headquarters");
        assertThat(((JsonObjectSchema) result.properties().get("company")).properties().keySet())
            .containsExactly("ticker", "name", "cik", "exchange");
        assertThat(((JsonObjectSchema) result.properties().get("headquarters")).properties().keySet())
            .containsExactly("street", "city", "country", "zip");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> properties(Map<String, Object> schema) {
        return (Map<String, Object>) schema.get("properties");
    }

    // Map.of does not keep insertion order, so the input schema is built from LinkedHashMaps, like a parsed flow
    private static Map<String, Object> ordered(Object... keysAndValues) {
        Map<String, Object> map = new LinkedHashMap<>();
        for (int i = 0; i < keysAndValues.length; i += 2) {
            map.put((String) keysAndValues[i], keysAndValues[i + 1]);
        }
        return map;
    }
}
