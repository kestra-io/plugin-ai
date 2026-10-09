package io.kestra.plugin.ai.provider;

import java.security.KeyPairGenerator;
import java.util.Base64;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.auth.oauth2.ImpersonatedCredentials;
import com.google.auth.oauth2.ServiceAccountCredentials;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.validations.ModelValidator;
import io.kestra.core.serializers.JacksonMapper;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class GoogleVertexAITest {
    @Inject
    private TestRunContextFactory runContextFactory;

    @Inject
    private ModelValidator modelValidator;

    @Test
    void shouldBeValidWithoutEndpoint() {
        var provider = GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue("gemini-3.5-flash-lite"))
            .location(Property.ofValue("us-central1"))
            .projectId(Property.ofValue("my-project"))
            .build();

        assertThat(modelValidator.isValid(provider)).isEmpty();
    }

    @Test
    void shouldAcceptLegacyProjectProperty() throws Exception {
        var runContext = runContextFactory.of(Map.of());
        var provider = JacksonMapper.ofJson().convertValue(
            Map.of(
                "type", GoogleVertexAI.class.getName(),
                "modelName", "gemini-3.5-flash-lite",
                "location", "us-central1",
                "project", "my-project"
            ),
            GoogleVertexAI.class
        );

        assertThat(runContext.render(provider.getProjectId()).as(String.class)).contains("my-project");
        assertThat(runContext.render(provider.getScopes()).asList(String.class)).containsExactly("https://www.googleapis.com/auth/cloud-platform");
    }

    @Test
    void connection_shouldUseServiceAccountAndInferProjectId() throws Exception {
        var runContext = runContextFactory.of(Map.of("serviceAccount", serviceAccountJson()));
        var provider = GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue("gemini-3.5-flash-lite"))
            .serviceAccount(Property.ofExpression("{{ serviceAccount }}"))
            .build();

        var connection = provider.connection(runContext);

        assertThat(connection.credentials()).isInstanceOf(ServiceAccountCredentials.class);
        assertThat(((ServiceAccountCredentials) connection.credentials()).getClientEmail()).isEqualTo("kestra@my-project.iam.gserviceaccount.com");
        assertThat(((ServiceAccountCredentials) connection.credentials()).getScopes()).containsExactly("https://www.googleapis.com/auth/cloud-platform");
        assertThat(runContext.render(connection.projectId()).as(String.class)).contains("my-project");
    }

    @Test
    void connection_shouldImpersonateServiceAccount() throws Exception {
        var provider = GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue("gemini-3.5-flash-lite"))
            .serviceAccount(Property.ofValue(serviceAccountJson()))
            .impersonatedServiceAccount(Property.ofValue("target@my-project.iam.gserviceaccount.com"))
            .build();

        var runContext = runContextFactory.of(Map.of());
        var connection = provider.connection(runContext);

        assertThat(connection.credentials()).isInstanceOf(ImpersonatedCredentials.class);
        assertThat(((ImpersonatedCredentials) connection.credentials()).getAccount()).isEqualTo("target@my-project.iam.gserviceaccount.com");
        assertThat(runContext.render(connection.projectId()).as(String.class)).contains("my-project");
    }

    @Test
    void imageModel_shouldRejectServiceAccount() throws Exception {
        var provider = GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue("imagen-3.0-generate-002"))
            .projectId(Property.ofValue("my-project"))
            .serviceAccount(Property.ofValue(serviceAccountJson()))
            .build();

        assertThatThrownBy(() -> provider.imageModel(runContextFactory.of(Map.of())))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("serviceAccount");
    }

    private static String serviceAccountJson() throws Exception {
        var keyPairGenerator = KeyPairGenerator.getInstance("RSA");
        keyPairGenerator.initialize(2048);
        var privateKey = "-----BEGIN PRIVATE KEY-----\n"
            + Base64.getMimeEncoder().encodeToString(keyPairGenerator.generateKeyPair().getPrivate().getEncoded())
            + "\n-----END PRIVATE KEY-----\n";

        return new ObjectMapper().writeValueAsString(
            Map.of(
                "type", "service_account",
                "project_id", "my-project",
                "private_key_id", "key-id",
                "private_key", privateKey,
                "client_email", "kestra@my-project.iam.gserviceaccount.com",
                "client_id", "123456789",
                "token_uri", "https://oauth2.googleapis.com/token"
            )
        );
    }
}
