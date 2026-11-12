package io.kestra.plugin.ai.tool.internal;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonBooleanSchema;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonNumberSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;

import static org.assertj.core.api.Assertions.assertThat;

class JsonObjectSchemaTranslatorTest {
    @Test
    void keepsPropertyOrderOfTopLevelNestedAndDefinitionSchemas() {
        var company = ordered(
            "type", "object",
            "properties", ordered(
                "ticker", ordered("type", "string"),
                "name", ordered("type", "string"),
                "cik", ordered("type", "integer"),
                "exchange", ordered("type", "string")
            )
        );
        var address = ordered(
            "type", "object",
            "properties", ordered(
                "street", ordered("type", "string"),
                "city", ordered("type", "string"),
                "country", ordered("type", "string"),
                "zip", ordered("type", "string")
            )
        );
        var schema = ordered(
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

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(result.properties().keySet())
            .containsExactly("event_summary", "sentiment", "zeta", "alpha", "company", "headquarters");
        assertThat(((JsonObjectSchema) result.properties().get("company")).properties().keySet())
            .containsExactly("ticker", "name", "cik", "exchange");
        assertThat(((JsonObjectSchema) result.properties().get("headquarters")).properties().keySet())
            .containsExactly("street", "city", "country", "zip");
    }

    @Test
    void descriptionTakesPrecedenceOverTitleForStringProperty() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "issued_on", ordered(
                    "type", "string",
                    "description", "Date of issue in DD/MM/YYYY format",
                    "title", "Short Title"
                )
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(result.properties().get("issued_on"))
            .isInstanceOf(JsonStringSchema.class)
            .extracting(e -> ((JsonStringSchema) e).description())
            .isEqualTo("Date of issue in DD/MM/YYYY format");
    }

    @Test
    void titleIsUsedAsFallbackWhenDescriptionAbsentForStringProperty() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "issued_on", ordered(
                    "type", "string",
                    "title", "Date of issue in DD/MM/YYYY format"
                )
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(result.properties().get("issued_on"))
            .isInstanceOf(JsonStringSchema.class)
            .extracting(e -> ((JsonStringSchema) e).description())
            .isEqualTo("Date of issue in DD/MM/YYYY format");
    }

    @Test
    void descriptionIsNullWhenNeitherDescriptionNorTitlePresent() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "issued_on", ordered("type", "string")
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(((JsonStringSchema) result.properties().get("issued_on")).description()).isNull();
    }

    @Test
    void descriptionTakesPrecedenceOverTitleForNumberProperty() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "score", ordered(
                    "type", "number",
                    "description", "Confidence score between 0.0 and 1.0",
                    "title", "Score"
                )
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(((JsonNumberSchema) result.properties().get("score")).description())
            .isEqualTo("Confidence score between 0.0 and 1.0");
    }

    @Test
    void titleFallbackWorksForNumberProperty() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "score", ordered("type", "number", "title", "Confidence Score")
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(((JsonNumberSchema) result.properties().get("score")).description())
            .isEqualTo("Confidence Score");
    }

    @Test
    void descriptionTakesPrecedenceOverTitleForIntegerProperty() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "count", ordered(
                    "type", "integer",
                    "description", "Total number of items",
                    "title", "Count"
                )
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(((JsonIntegerSchema) result.properties().get("count")).description())
            .isEqualTo("Total number of items");
    }

    @Test
    void descriptionTakesPrecedenceOverTitleForBooleanProperty() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "active", ordered(
                    "type", "boolean",
                    "description", "True if the account is active",
                    "title", "Active"
                )
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(((JsonBooleanSchema) result.properties().get("active")).description())
            .isEqualTo("True if the account is active");
    }

    @Test
    void descriptionTakesPrecedenceOverTitleForEnumProperty() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "status", ordered(
                    "type", "string",
                    "description", "Current lifecycle status of the entity",
                    "title", "Status",
                    "enum", List.of("ACTIVE", "INACTIVE", "PENDING")
                )
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(result.properties().get("status")).isInstanceOf(JsonEnumSchema.class);
        assertThat(((JsonEnumSchema) result.properties().get("status")).description())
            .isEqualTo("Current lifecycle status of the entity");
        assertThat(((JsonEnumSchema) result.properties().get("status")).enumValues())
            .containsExactly("ACTIVE", "INACTIVE", "PENDING");
    }

    @Test
    void titleFallbackWorksForEnumProperty() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "status", ordered(
                    "type", "string",
                    "title", "Lifecycle Status",
                    "enum", List.of("ACTIVE", "INACTIVE")
                )
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(((JsonEnumSchema) result.properties().get("status")).description())
            .isEqualTo("Lifecycle Status");
    }

    @Test
    void descriptionTakesPrecedenceOverTitleForArrayProperty() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "tags", ordered(
                    "type", "array",
                    "description", "List of lowercase tag strings",
                    "title", "Tags",
                    "items", ordered("type", "string", "description", "A single lowercase tag")
                )
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        var tagsSchema = (JsonArraySchema) result.properties().get("tags");
        assertThat(tagsSchema.description()).isEqualTo("List of lowercase tag strings");
        assertThat(((JsonStringSchema) tagsSchema.items()).description()).isEqualTo("A single lowercase tag");
    }

    @Test
    void descriptionTakesPrecedenceOverTitleForNestedObjectProperty() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "address", ordered(
                    "type", "object",
                    "description", "Mailing address of the entity",
                    "title", "Address",
                    "properties", ordered(
                        "city", ordered("type", "string", "description", "City name in English")
                    )
                )
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        var addressSchema = (JsonObjectSchema) result.properties().get("address");
        assertThat(addressSchema.description()).isEqualTo("Mailing address of the entity");
        assertThat(((JsonStringSchema) addressSchema.properties().get("city")).description())
            .isEqualTo("City name in English");
    }

    @Test
    void rootSchemaDescriptionFallsBackToSchemaDescriptionFieldWhenCallerPassesNull() {
        var schema = ordered(
            "type", "object",
            "description", "Invoice extraction schema",
            "properties", ordered(
                "amount", ordered("type", "number")
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(result.description()).isEqualTo("Invoice extraction schema");
    }

    @Test
    void rootSchemaTitleUsedWhenDescriptionAbsentAndCallerPassesNull() {
        var schema = ordered(
            "type", "object",
            "title", "Invoice Schema",
            "properties", ordered(
                "amount", ordered("type", "number")
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(result.description()).isEqualTo("Invoice Schema");
    }

    @Test
    void callerExplicitDescriptionAlwaysTakesPrecedenceOverSchemaDescription() {
        var schema = ordered(
            "type", "object",
            "description", "Schema-level description",
            "title", "Schema Title",
            "properties", ordered(
                "amount", ordered("type", "number")
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, "Caller-provided description");

        assertThat(result.description()).isEqualTo("Caller-provided description");
    }

    @Test
    void descriptionIsAttachedWhenTypeOmitted() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "untyped_field", ordered(
                    "description", "Instruction for untyped field",
                    "title", "Untyped"
                )
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(result.properties().get("untyped_field"))
            .isInstanceOf(JsonStringSchema.class)
            .extracting(e -> ((JsonStringSchema) e).description())
            .isEqualTo("Instruction for untyped field");
    }

    @Test
    void descriptionAppliedToAnyOfWhenSubSchemaHasNone() {
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "age", ordered(
                    "description", "User age in years",
                    "anyOf", List.of(
                        ordered("type", "integer"),
                        ordered("type", "null")
                    )
                )
            )
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        assertThat(result.properties().get("age"))
            .isInstanceOf(JsonIntegerSchema.class)
            .extracting(e -> ((JsonIntegerSchema) e).description())
            .isEqualTo("User age in years");
    }

    @Test
    void siblingDescriptionAppliedToRefWhenDefinitionHasNone() {
        var address = ordered(
            "type", "object",
            "properties", ordered(
                "street", ordered("type", "string")
            )
        );
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "headquarters", ordered(
                    "$ref", "#/$defs/address",
                    "description", "Main corporate office"
                )
            ),
            "$defs", ordered("address", address)
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        var hq = (JsonObjectSchema) result.properties().get("headquarters");
        assertThat(hq.description()).isEqualTo("Main corporate office");
    }

    @Test
    void fieldDescriptionOverridesDefinitionTitle() {
        var address = ordered(
            "type", "object",
            "title", "Address",
            "properties", ordered(
                "street", ordered("type", "string")
            )
        );
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "headquarters", ordered(
                    "$ref", "#/$defs/address",
                    "description", "Main corporate office"
                )
            ),
            "$defs", ordered("address", address)
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        var hq = (JsonObjectSchema) result.properties().get("headquarters");
        assertThat(hq.description()).isEqualTo("Main corporate office");
    }

    @Test
    void fieldDescriptionOverridesDefinitionTitleViaAnyOf() {
        var address = ordered(
            "type", "object",
            "title", "Address",
            "properties", ordered(
                "street", ordered("type", "string")
            )
        );
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "headquarters", ordered(
                    "description", "Main corporate office",
                    "anyOf", List.of(
                        ordered("$ref", "#/$defs/address"),
                        ordered("type", "null")
                    )
                )
            ),
            "$defs", ordered("address", address)
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        var hq = (JsonObjectSchema) result.properties().get("headquarters");
        assertThat(hq.description()).isEqualTo("Main corporate office");
    }

    @Test
    void definitionDescriptionWinsOverFieldTitle() {
        var address = ordered(
            "type", "object",
            "title", "Address",
            "description", "Mailing address of the entity",
            "properties", ordered(
                "street", ordered("type", "string")
            )
        );
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "headquarters", ordered(
                    "$ref", "#/$defs/address",
                    "title", "Headquarters"
                )
            ),
            "$defs", ordered("address", address)
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        var hq = (JsonObjectSchema) result.properties().get("headquarters");
        assertThat(hq.description()).isEqualTo("Mailing address of the entity");
    }

    @Test
    void definitionDescriptionWinsOverFieldTitleViaAnyOf() {
        var address = ordered(
            "type", "object",
            "title", "Address",
            "description", "Mailing address of the entity",
            "properties", ordered(
                "street", ordered("type", "string")
            )
        );
        var schema = ordered(
            "type", "object",
            "properties", ordered(
                "headquarters", ordered(
                    "title", "Headquarters",
                    "anyOf", List.of(
                        ordered("$ref", "#/$defs/address"),
                        ordered("type", "null")
                    )
                )
            ),
            "$defs", ordered("address", address)
        );

        var result = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, null);

        var hq = (JsonObjectSchema) result.properties().get("headquarters");
        assertThat(hq.description()).isEqualTo("Mailing address of the entity");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> properties(Map<String, Object> schema) {
        return (Map<String, Object>) schema.get("properties");
    }

    // Map.of does not keep insertion order, so the input schema is built from LinkedHashMaps, like a parsed flow
    private static Map<String, Object> ordered(Object... keysAndValues) {
        var map = new LinkedHashMap<String, Object>();
        for (int i = 0; i < keysAndValues.length; i += 2) {
            map.put((String) keysAndValues[i], keysAndValues[i + 1]);
        }
        return map;
    }
}
