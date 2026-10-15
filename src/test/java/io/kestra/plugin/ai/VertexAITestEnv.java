package io.kestra.plugin.ai;

import java.util.Optional;

import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.provider.GoogleVertexAI;

/**
 * Live Vertex AI tests run whenever Application Default Credentials are available ({@value #CREDENTIALS_ENV}).
 * On CI, the shared plugins workflow sets them from the {@code GCP_SERVICE_ACCOUNT} secret, which is granted
 * {@code roles/aiplatform.user} on the {@code kestra-prd} project. The secret isn't exposed to forks, so the tests are skipped there.
 * Locally, export {@value #CREDENTIALS_ENV} and override {@code VERTEX_AI_PROJECT} / {@code VERTEX_AI_LOCATION} if needed.
 */
public final class VertexAITestEnv {
    public static final String CREDENTIALS_ENV = "GOOGLE_APPLICATION_CREDENTIALS";
    public static final String PROJECT = Optional.ofNullable(System.getenv("VERTEX_AI_PROJECT")).orElse("kestra-prd");
    public static final String LOCATION = Optional.ofNullable(System.getenv("VERTEX_AI_LOCATION")).orElse("global");
    // Embedding models need a regional location: langchain4j counts tokens before embedding, and `countTokens` isn't served on `global`
    public static final String EMBEDDING_LOCATION = Optional.ofNullable(System.getenv("VERTEX_AI_EMBEDDING_LOCATION")).orElse("europe-west1");
    public static final String CHAT_MODEL = "gemini-3.5-flash-lite";
    // Gemini 3 models require thought signatures on function calls, which langchain4j-vertex-ai-gemini doesn't send back
    public static final String TOOL_CHAT_MODEL = "gemini-2.5-flash";
    public static final String EMBEDDING_MODEL = "gemini-embedding-001";

    private VertexAITestEnv() {
    }

    public static GoogleVertexAI provider(String modelName) {
        return provider(modelName, LOCATION);
    }

    public static GoogleVertexAI embeddingProvider() {
        return provider(EMBEDDING_MODEL, EMBEDDING_LOCATION);
    }

    private static GoogleVertexAI provider(String modelName, String location) {
        return GoogleVertexAI.builder()
            .type(GoogleVertexAI.class.getName())
            .modelName(Property.ofValue(modelName))
            .projectId(Property.ofValue(PROJECT))
            .location(Property.ofValue(location))
            .build();
    }
}
