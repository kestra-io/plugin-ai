package io.kestra.plugin.ai.provider.github;

import java.util.*;

/**
 * Adapter between Kestra plugin AI tasks and GitHubModelsClient.
 *
 * IMPORTANT:
 * - This class is a neutral adapter. If your project requires implementation of a specific
 *   provider interface (e.g., ChatProvider), you'll need to adapt method signatures accordingly.
 * - The methods below are simple and designed to be easy to call from task code.
 */
public class GitHubProvider {
    private final GitHubModelsConfig config;
    private final GitHubModelsClient client;

    public GitHubProvider(GitHubModelsConfig config) {
        this.config = config;
        this.client = new GitHubModelsClient(config);
    }

    /**
     * Send a chat prompt and return the assistant reply as a String.
     *
     * @param model  model id (or null to use config.defaultModel)
     * @param prompt user prompt text
     * @return assistant reply text
     * @throws Exception on errors
     */
    public String chat(String model, String prompt) throws Exception {
        // Build a minimal messages list (user role)
        Map<String, String> userMsg = Map.of(
            "role", "user",
            "content", prompt
        );
        List<Map<String, String>> messages = Collections.singletonList(userMsg);

        // No extra params for now; can be extended to pass temperature, max_tokens, etc.
        return client.chat(model != null ? model : config.getDefaultModel(), messages, null, null);
    }

    /**
     * Send pre-constructed messages (for more advanced use).
     *
     * @param model    model id
     * @param messages list of message maps with keys "role" and "content"
     * @param params   extra parameters for the model call
     * @return assistant text
     * @throws Exception on errors
     */
    public String chatWithMessages(String model, List<Map<String, String>> messages, Map<String, Object> params) throws Exception {
        return client.chat(model != null ? model : config.getDefaultModel(), messages, null, params);
    }

    /**
     * Get embeddings for an input text.
     *
     * @param model model id for embeddings
     * @param input text to embed
     * @return embedding vector
     * @throws Exception on errors
     */
    public List<Double> embeddings(String model, String input) throws Exception {
        return client.embeddings(model != null ? model : config.getDefaultModel(), input);
    }

    /**
     * Expose config for tests or consumers.
     */
    public GitHubModelsConfig getConfig() {
        return this.config;
    }
}
