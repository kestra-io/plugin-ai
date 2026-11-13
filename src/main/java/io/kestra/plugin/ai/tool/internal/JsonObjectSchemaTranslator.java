package io.kestra.plugin.ai.tool.internal;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import io.kestra.core.utils.MapUtils;

import dev.langchain4j.model.chat.request.json.*;

public final class JsonObjectSchemaTranslator {
    private JsonObjectSchemaTranslator() {
        // utility class pattern
    }

    @SuppressWarnings("unchecked")
    public static JsonObjectSchema fromOpenAPISchema(Map<String, Object> schema, String description) {
        var definitions = mapDefinitions((Map<String, Object>) schema.get("$defs"));
        definitions = replaceDefinitions(definitions, definitions);
        var properties = mapProperties((Map<String, Object>) schema.get("properties"), definitions);
        properties = replaceDefinitions(properties, definitions);

        var effectiveDescription = description != null ? description : descriptionOf(schema);

        return JsonObjectSchema.builder()
            .addProperties(properties)
            .required((List<String>) schema.get("required"))
            .description(effectiveDescription)
            .build();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, JsonSchemaElement> mapDefinitions(Map<String, Object> definitionSchemas) {
        return MapUtils.emptyOnNull(definitionSchemas).entrySet().stream()
            .map(entry ->
            {
                var schema = (Map<String, Object>) entry.getValue();
                var jsonSchemaElement = fromOpenAPISchema(schema, null);
                return Map.entry(
                    entry.getKey(),
                    jsonSchemaElement
                );
            })
            .collect(toOrderedMap());
    }

    @SuppressWarnings("unchecked")
    private static Map<String, JsonSchemaElement> mapProperties(Map<String, Object> properties, Map<String, JsonSchemaElement> definitions) {
        return MapUtils.emptyOnNull(properties).entrySet().stream()
            .filter(prop -> !prop.getKey().equals("type"))
            .map(entry ->
            {
                var schema = (Map<String, Object>) entry.getValue();
                var jsonSchemaElement = mapSchema(schema, definitions);
                return Map.entry(
                    entry.getKey(),
                    jsonSchemaElement
                );
            })
            .filter(entry -> !(entry.getValue() instanceof JsonNullSchema))
            .collect(toOrderedMap());
    }

    private static String descriptionOf(Map<String, Object> schema) {
        if (schema == null) {
            return null;
        }
        var description = (String) schema.get("description");
        return description != null ? description : (String) schema.get("title");
    }

    @SuppressWarnings("unchecked")
    private static JsonSchemaElement mapSchema(Map<String, Object> schema, Map<String, JsonSchemaElement> definitions) {
        if (schema == null) {
            return new JsonStringSchema();
        }

        var type = (String) schema.get("type");
        // JSON Schema "description" carries field instructions; "title" is only a label.
        var description = descriptionOf(schema);
        var _enum = (List<String>) schema.get("enum");
        var anyOf = (List<Map<String, Object>>) schema.get("anyOf");
        var ref = (String) schema.get("$ref");
        if (_enum != null) {
            return JsonEnumSchema.builder().description(description).enumValues(_enum).build();
        }
        if (anyOf != null) {
            // anyOf is not supported by all LLM (for ex gemini).
            // So instead, we try to find a type which is not a string and return it as it's supposed to be more specific.
            return anyOf.stream()
                .filter(subSchema -> !"string".equals(subSchema.get("type")) && !"null".equals(subSchema.get("type")))
                .map(subSchema -> mapSchema(subSchema, definitions))
                .findAny()
                .map(element -> schema.get("description") != null ? withDescription(element, description) : applyDescriptionIfAbsent(element, description))
                .orElse(JsonStringSchema.builder().description(description).build());
        }
        if (ref != null) {
            // reference starts with "#/$defs/"
            var referenceType = ref.substring(8);
            if (definitions != null && definitions.containsKey(referenceType)) {
                var resolved = definitions.get(referenceType);
                return schema.get("description") != null ? withDescription(resolved, description) : applyDescriptionIfAbsent(resolved, description);
            }
            return JsonReferenceSchema.builder().reference(referenceType).build();
        }

        return switch (type) {
            case "number" -> JsonNumberSchema.builder().description(description).build();
            case "integer" -> JsonIntegerSchema.builder().description(description).build();
            case "boolean" -> JsonBooleanSchema.builder().description(description).build();
            case "null" -> new JsonNullSchema();
            case "object" -> JsonObjectSchema.builder()
                .description(description)
                .required((List<String>) schema.get("required"))
                .addProperties(mapProperties((Map<String, Object>) schema.get("properties"), definitions))
                .build();
            case "array" ->
                JsonArraySchema.builder()
                    .description(description)
                    .items(mapSchema((Map<String, Object>) schema.get("items"), definitions))
                    .build();
            // it should not happen, but in this case we use String as it's the most proficient type for an LLM
            case null -> JsonStringSchema.builder().description(description).build();
            // we coalesce other types to String for now...
            default -> JsonStringSchema.builder().description(description).build();
        };
    }

    private static JsonSchemaElement applyDescriptionIfAbsent(JsonSchemaElement element, String description) {
        if (description == null || element == null || element.description() != null) {
            return element;
        }
        return withDescription(element, description);
    }

    private static JsonSchemaElement withDescription(JsonSchemaElement element, String description) {
        if (element == null) {
            return null;
        }
        return switch (element) {
            case JsonStringSchema s -> JsonStringSchema.builder().description(description).build();
            case JsonIntegerSchema s -> JsonIntegerSchema.builder().description(description).build();
            case JsonNumberSchema s -> JsonNumberSchema.builder().description(description).build();
            case JsonBooleanSchema s -> JsonBooleanSchema.builder().description(description).build();
            case JsonEnumSchema s -> JsonEnumSchema.builder().description(description).enumValues(s.enumValues()).build();
            case JsonArraySchema s -> JsonArraySchema.builder().description(description).items(s.items()).build();
            case JsonObjectSchema s -> JsonObjectSchema.builder()
                .description(description)
                .required(s.required())
                .addProperties(s.properties())
                .build();
            default -> element;
        };
    }

    private static Map<String, JsonSchemaElement> replaceDefinitions(Map<String, JsonSchemaElement> properties, Map<String, JsonSchemaElement> definitions) {
        return properties.entrySet().stream()
            .map(entry ->
            {
                JsonSchemaElement schema = replaceDefinitions(entry.getValue(), definitions);
                return Map.entry(entry.getKey(), schema);
            })
            .collect(toOrderedMap());
    }

    // LinkedHashMap keeps the property order of the flow, which is the order the model generates the fields in
    private static <V> Collector<Map.Entry<String, ? extends V>, ?, Map<String, V>> toOrderedMap() {
        return Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (first, second) -> first, LinkedHashMap::new);
    }

    private static JsonSchemaElement replaceDefinitions(JsonSchemaElement schema, Map<String, JsonSchemaElement> definitions) {
        if (schema == null) {
            return null;
        }

        return switch (schema) {
            case JsonReferenceSchema jsonReferenceSchema ->
                definitions.getOrDefault(
                    jsonReferenceSchema.reference(),
                    JsonObjectSchema.builder().description(jsonReferenceSchema.description()).build()
                );
            case JsonObjectSchema jsonObjectSchema -> JsonObjectSchema.builder()
                .description(jsonObjectSchema.description())
                .required(jsonObjectSchema.required())
                .addProperties(replaceDefinitions(jsonObjectSchema.properties(), definitions))
                .build();
            case JsonArraySchema jsonArraySchema -> JsonArraySchema.builder()
                .description(jsonArraySchema.description())
                .items(replaceDefinitions(jsonArraySchema.items(), definitions))
                .build();
            default -> schema;
        };
    }
}
