package io.kestra.plugin.ai.tool.internal;

import dev.langchain4j.model.chat.request.json.*;
import io.kestra.core.utils.MapUtils;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public final class JsonObjectSchemaTranslator {
    private JsonObjectSchemaTranslator() {
        // utility class pattern
    }

    @SuppressWarnings("unchecked")
    public static JsonObjectSchema fromOpenAPISchema(Map<String, Object> schema, String description) {
        var definitions = mapDefinitions((Map<String, Object>) schema.get("$defs"));
        var properties = mapProperties((Map<String, Object>) schema.get("properties"));

        // some LLM didn't support definitions, so we will remove them and replace them by their schema.
        definitions = replaceDefinitions(definitions, definitions);
        properties = replaceDefinitions(properties, definitions);

        return JsonObjectSchema.builder()
            .addProperties(properties)
            .required((List<String>) schema.get("required"))
            .description(description)
            .build();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, JsonSchemaElement> mapDefinitions(Map<String, Object> definitionSchemas) {
        return MapUtils.emptyOnNull(definitionSchemas).entrySet().stream()
            .map(entry -> {
                var schema = (Map<String, Object>) entry.getValue();
                var jsonSchemaElement = fromOpenAPISchema(schema, null);
                return Map.entry(
                    entry.getKey(),
                    jsonSchemaElement
                );
            })
            .collect(Collectors.toMap(
                entry -> entry.getKey(),
                entry -> entry.getValue()
            ));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, JsonSchemaElement> mapProperties(Map<String, Object> properties) {
        return MapUtils.emptyOnNull(properties).entrySet().stream()
            .filter(prop -> !prop.getKey().equals("type"))
            .map(entry -> {
                var schema = (Map<String, Object>) entry.getValue();
                var jsonSchemaElement = mapSchema(schema);
                return Map.entry(
                    entry.getKey(),
                    jsonSchemaElement
                );
            })
            .filter(entry -> !(entry.getValue() instanceof JsonNullSchema))
            .collect(Collectors.toMap(
                entry -> entry.getKey(),
                entry -> entry.getValue()
            ));
    }

    @SuppressWarnings("unchecked")
    private static JsonSchemaElement mapSchema(Map<String, Object> schema) {
        if (schema == null) {
            return new JsonStringSchema();
        }

        var type = (String) schema.get("type");
        var title = (String) schema.get("title");
        var _enum = (List<String>) schema.get("enum");
        var anyOf = (List<Map<String, Object>>) schema.get("anyOf");
        var ref = (String) schema.get("$ref");
        if (_enum != null) {
            return JsonEnumSchema.builder().description(title).enumValues(_enum).build();
        }
        if (anyOf != null) {
            // anyOf is not supported by all LLM (for ex gemini).
            // So instead, we try to find a type which is not a string and return it as it's supposed to be more specific.
            return anyOf.stream()
                .filter(subSchema -> !"string".equals(subSchema.get("type")))
                .map(subSchema -> mapSchema(subSchema))
                .findAny()
                .orElse(JsonStringSchema.builder().description(title).build());
        }
        if (ref != null) {
            // reference starts with "#/$defs/"
            String referenceType = ref.substring(8);
            return JsonReferenceSchema.builder().reference(referenceType).build();
        }

        return switch (type) {
            case "number" -> JsonNumberSchema.builder().description(title).build();
            case "integer" -> JsonIntegerSchema.builder().description(title).build();
            case "boolean" -> JsonBooleanSchema.builder().description(title).build();
            case "null" -> new JsonNullSchema();
            case "object" -> JsonObjectSchema.builder()
                .description(title)
                .required((List<String>) schema.get("required"))
                .addProperties(mapProperties((Map<String, Object>) schema.get("properties")))
                .build();
            case "array" ->
                JsonArraySchema.builder()
                    .description(title)
                    .items(mapSchema((Map<String, Object>) schema.get("items")))
                    .build();
            // it should not happen, but in this case we use String as it's the most proficient type for an LLM
            case null -> new JsonStringSchema();
            // we coalesce other types to String for now...
            default -> JsonStringSchema.builder().description(title).build();
        };
    }

    private static Map<String, JsonSchemaElement> replaceDefinitions(Map<String, JsonSchemaElement> properties, Map<String, JsonSchemaElement> definitions) {
        return properties.entrySet().stream()
            .map(entry -> {
                JsonSchemaElement schema = replaceDefinitions(entry.getValue(), definitions);
                return Map.entry(entry.getKey(), schema);
            })
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    private static JsonSchemaElement replaceDefinitions(JsonSchemaElement schema, Map<String, JsonSchemaElement> definitions) {
        if (schema == null) {
            return null;
        }

        return switch (schema) {
            case JsonReferenceSchema jsonReferenceSchema ->
                definitions.getOrDefault(jsonReferenceSchema.reference(),
                    JsonObjectSchema.builder().description(jsonReferenceSchema.description()).build());
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
