package io.kestra.plugin.ai.provider.github;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.*;

/**
 * Minimal GitHub Models client (chat + embeddings).
 *
 * Notes:
 * - Uses the base URL from GitHubModelsConfig (defaults to https://api.github.com).
 * - Supports chat with `messages` (preferred) or `input`.
 * - Parses typical response shapes: choices[0].message.content, choices[0].output, data[].embedding.
 */
public class GitHubModelsClient {
    private final GitHubModelsConfig config;
    private final HttpClient http;
    private final ObjectMapper mapper = new ObjectMapper();

    public GitHubModelsClient(GitHubModelsConfig config) {
        this.config = config;
        int timeout = config.getTimeoutMs() != null ? config.getTimeoutMs() : 60_000;
        this.http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofMillis(timeout))
            .build();
    }

    private String base() {
        return (config.getBaseUrl() != null && !config.getBaseUrl().isBlank())
            ? config.getBaseUrl()
            : "https://api.github.com";
    }

    /**
     * Call GitHub chat completion endpoint.
     *
     * @param model    model id (e.g. "gpt-4.1-mini")
     * @param messages list of message maps (role, content) OR null if using input
     * @param input    fallback input string (used if messages == null)
     * @param params   optional additional parameters (temperature, max_tokens, etc.)
     * @return assistant textual response (best-effort parsing)
     * @throws Exception on network / parsing errors
     */
    public String chat(String model, List<Map<String, String>> messages, String input, Map<String, Object> params) throws Exception {
        String url = base() + "/inference/chat/completions";

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("model", model != null ? model : config.getDefaultModel());
        if (messages != null) {
            payload.put("messages", messages);
        } else {
            payload.put("input", input != null ? input : "");
        }

        if (params != null) {
            payload.putAll(params);
        }

        HttpRequest req = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .timeout(Duration.ofMillis(config.getTimeoutMs() != null ? config.getTimeoutMs() : 60_000))
            .header("Authorization", "Bearer " + config.getToken())
            .header("Accept", "application/vnd.github+json")
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(payload)))
            .build();

        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        int status = resp.statusCode();
        String body = resp.body();

        if (status < 200 || status >= 300) {
            throw new RuntimeException("GitHub Models API error: " + status + " -> " + body);
        }

        JsonNode root = mapper.readTree(body);

        // 1) Try choices[0].message.content
        JsonNode choices = root.path("choices");
        if (choices.isArray() && choices.size() > 0) {
            JsonNode first = choices.get(0);
            JsonNode message = first.path("message");
            if (message.isObject() && message.has("content")) {
                return message.get("content").asText();
            }

            // 2) fallback to output field on choice
            if (first.has("output")) {
                return first.get("output").asText();
            }
        }

        // 3) fallback to top-level output
        if (root.has("output")) {
            return root.get("output").asText();
        }

        // 4) fallback: return whole body
        return root.toString();
    }

    /**
     * Call GitHub embeddings endpoint and return the first embedding vector as List<Double>.
     *
     * @param model model id for embeddings (e.g. "text-embedding-3-large")
     * @param input text to embed
     * @return embedding vector as List<Double>
     * @throws Exception on network / parsing errors
     */
    public List<Double> embeddings(String model, String input) throws Exception {
        String url = base() + "/inference/embeddings";

        Map<String, Object> payload = Map.of(
            "model", model != null ? model : config.getDefaultModel(),
            "input", input
        );

        HttpRequest req = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .timeout(Duration.ofMillis(config.getTimeoutMs() != null ? config.getTimeoutMs() : 60_000))
            .header("Authorization", "Bearer " + config.getToken())
            .header("Accept", "application/vnd.github+json")
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(payload)))
            .build();

        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        int status = resp.statusCode();
        String body = resp.body();

        if (status < 200 || status >= 300) {
            throw new RuntimeException("GitHub Models API embeddings error: " + status + " -> " + body);
        }

        JsonNode root = mapper.readTree(body);
        JsonNode data = root.path("data");
        if (data.isArray() && data.size() > 0) {
            JsonNode first = data.get(0);
            JsonNode emb = first.path("embedding");
            if (emb.isArray()) {
                List<Double> vec = new ArrayList<>();
                for (JsonNode n : emb) {
                    vec.add(n.asDouble());
                }
                return vec;
            }
        }

        throw new RuntimeException("No embedding found in response: " + body);
    }
}
