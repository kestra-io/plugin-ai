package io.kestra.plugin.ai.provider;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.validations.ModelValidator;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class GoogleVertexAITest {
    @Inject
    private ModelValidator modelValidator;

    @Test
    void shouldBeValidWithoutEndpoint() {
        var provider = GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue("gemini-3.5-flash-lite"))
            .location(Property.ofValue("us-central1"))
            .project(Property.ofValue("my-project"))
            .build();

        assertThat(modelValidator.isValid(provider)).isEmpty();
    }
}
