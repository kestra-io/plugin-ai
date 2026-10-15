package io.kestra.plugin.ai.provider;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.KeyPairGenerator;
import java.util.Base64;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.api.parallel.ResourceLock;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.auth.oauth2.ImpersonatedCredentials;
import com.google.auth.oauth2.ServiceAccountCredentials;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.validations.ModelValidator;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.ai.VertexAITestEnv;
import io.kestra.plugin.ai.domain.ChatConfiguration;

import dev.langchain4j.exception.RateLimitException;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.abort;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

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
    void shouldPreferProjectIdOverLegacyProjectProperty() throws Exception {
        var runContext = runContextFactory.of(Map.of());
        var provider = JacksonMapper.ofJson().convertValue(
            Map.of(
                "type", GoogleVertexAI.class.getName(),
                "modelName", "gemini-3.5-flash-lite",
                "location", "us-central1",
                "project", "legacy-project",
                "projectId", "my-project"
            ),
            GoogleVertexAI.class
        );

        assertThat(runContext.render(provider.getProjectId()).as(String.class)).contains("my-project");
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

    @Test
    void chatModel_shouldUseGlobalEndpointForGlobalLocation() throws Exception {
        var provider = GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue("gemini-2.5-pro"))
            .location(Property.ofValue("global"))
            .serviceAccount(Property.ofValue(serviceAccountJson()))
            .build();

        var chatModel = provider.chatModel(runContextFactory.of(Map.of()), ChatConfiguration.empty());

        assertThat(chatModel).extracting("vertexAI.apiEndpoint").isEqualTo("aiplatform.googleapis.com");
    }

    @Test
    void chatModel_shouldUseRegionalEndpointForRegionalLocation() throws Exception {
        var provider = GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue("gemini-2.5-pro"))
            .location(Property.ofValue("us-central1"))
            .serviceAccount(Property.ofValue(serviceAccountJson()))
            .build();

        var chatModel = provider.chatModel(runContextFactory.of(Map.of()), ChatConfiguration.empty());

        assertThat(chatModel).extracting("vertexAI.apiEndpoint").isEqualTo("us-central1-aiplatform.googleapis.com");
    }

    @Test
    void embeddingModel_shouldUseGlobalEndpointForGlobalLocation() throws Exception {
        var provider = GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue("gemini-embedding-001"))
            .location(Property.ofValue("global"))
            .serviceAccount(Property.ofValue(serviceAccountJson()))
            .build();

        var embeddingModel = provider.embeddingModel(runContextFactory.of(Map.of()));

        assertThat(embeddingModel).extracting("settings.endpoint").isEqualTo("aiplatform.googleapis.com:443");
        assertThat(embeddingModel).extracting("endpointName.publisher").isEqualTo("google");
    }

    @Test
    void embeddingModel_shouldPreferExplicitEndpointOverGlobalLocation() throws Exception {
        var provider = GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue("gemini-embedding-001"))
            .location(Property.ofValue("global"))
            .endpoint(Property.ofValue("europe-west1-aiplatform.googleapis.com:443"))
            .serviceAccount(Property.ofValue(serviceAccountJson()))
            .build();

        var embeddingModel = provider.embeddingModel(runContextFactory.of(Map.of()));

        assertThat(embeddingModel).extracting("settings.endpoint").isEqualTo("europe-west1-aiplatform.googleapis.com:443");
    }

    @Test
    void endpoint_shouldResolveForImageAndEmbeddingModels() throws Exception {
        var runContext = runContextFactory.of(Map.of());
        var provider = GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue("imagen-3.0-generate-002"))
            .build();
        var providerWithEndpoint = GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue("imagen-3.0-generate-002"))
            .endpoint(Property.ofValue("europe-west1-aiplatform.googleapis.com:443"))
            .build();

        assertThat(provider.endpoint(runContext, "global")).isEqualTo("aiplatform.googleapis.com:443");
        assertThat(provider.endpoint(runContext, "us-central1")).isNull();
        assertThat(providerWithEndpoint.endpoint(runContext, "global")).isEqualTo("europe-west1-aiplatform.googleapis.com:443");
    }

    @Test
    void isGlobal_shouldIgnoreCase() {
        assertThat(GoogleVertexAI.isGlobal("global")).isTrue();
        assertThat(GoogleVertexAI.isGlobal("GLOBAL")).isTrue();
        assertThat(GoogleVertexAI.isGlobal("us-central1")).isFalse();
        assertThat(GoogleVertexAI.isGlobal(null)).isFalse();
    }

    @Test
    @EnabledIfEnvironmentVariable(named = VertexAITestEnv.CREDENTIALS_ENV, matches = ".+")
    void chatModel_shouldCallVertexAIWithServiceAccount() throws Exception {
        var provider = serviceAccountProvider(VertexAITestEnv.CHAT_MODEL, VertexAITestEnv.LOCATION);

        var chatModel = provider.chatModel(runContextFactory.of(Map.of()), ChatConfiguration.empty());

        try {
            assertThat(chatModel.chat("Reply with the single word: pong")).containsIgnoringCase("pong");
        } catch (RateLimitException e) {
            abort("Skipped: Vertex AI rate limited (429)");
        }
    }

    @Test
    @EnabledIfEnvironmentVariable(named = VertexAITestEnv.CREDENTIALS_ENV, matches = ".+")
    void embeddingModel_shouldCallVertexAIWithServiceAccount() throws Exception {
        var provider = serviceAccountProvider(VertexAITestEnv.EMBEDDING_MODEL, VertexAITestEnv.EMBEDDING_LOCATION);

        var embeddingModel = provider.embeddingModel(runContextFactory.of(Map.of()));

        try {
            assertThat(embeddingModel.embed("pong").content().vector()).isNotEmpty();
        } catch (RateLimitException e) {
            abort("Skipped: Vertex AI rate limited (429)");
        }
    }

    // Live calls through the `serviceAccount` property rather than ADC, reusing the ADC key file when it is a service account key
    // (on CI, the GCP_SERVICE_ACCOUNT secret). Locally, ADC is usually user credentials, which the property doesn't accept.
    private static GoogleVertexAI serviceAccountProvider(String modelName, String location) throws Exception {
        var credentials = Files.readString(Path.of(System.getenv(VertexAITestEnv.CREDENTIALS_ENV)));
        assumeTrue(
            "service_account".equals(JacksonMapper.ofJson().readTree(credentials).path("type").asText()),
            "Application Default Credentials are not a service account key"
        );

        return GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue(modelName))
            .projectId(Property.ofValue(VertexAITestEnv.PROJECT))
            .location(Property.ofValue(location))
            .serviceAccount(Property.ofValue(credentials))
            .build();
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
