package io.kestra.plugin.ai.langdock;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.kestra.core.http.HttpRequest;
import io.kestra.core.http.HttpResponse;
import io.kestra.core.http.client.HttpClient;
import io.kestra.core.http.client.HttpClientResponseException;
import io.kestra.core.http.client.configurations.HttpConfiguration;
import io.kestra.core.http.client.configurations.TimeoutConfiguration;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "List available Langdock models",
    description = """
        Calls Langdock's `GET /models` endpoint on the route selected by `modelFamily`, to discover the \
        model IDs available to your workspace. Langdock does not publish a stable response schema for \
        this endpoint, so the response is parsed leniently: only `id` is required per model, and \
        provider-specific fields (`created`/`owned_by` on the OpenAI route, `display_name`/`created_at` \
        on the Anthropic route) are kept when present."""
)
@Plugin(
    examples = {
        @Example(
            title = "List Langdock OpenAI-route models and use the first one in a chat completion",
            full = true,
            code = {
                """
                    id: list_and_chat
                    namespace: company.ai

                    tasks:
                      - id: list_models
                        type: io.kestra.plugin.ai.langdock.ListModels
                        apiKey: "{{ secret('LANGDOCK_API_KEY') }}"
                        modelFamily: OPENAI

                      - id: chat_completion
                        type: io.kestra.plugin.ai.completion.ChatCompletion
                        provider:
                          type: io.kestra.plugin.ai.provider.Langdock
                          apiKey: "{{ secret('LANGDOCK_API_KEY') }}"
                          modelFamily: OPENAI
                          modelName: "{{ outputs.list_models.models[0].id }}"
                        messages:
                          - type: USER
                            content: Say hello in one short sentence.
                    """
            }
        )
    }
)
public class ListModels extends Task implements RunnableTask<ListModels.Output> {
    private static final ObjectMapper MAPPER = JacksonMapper.ofJson();
    private static final int TRUNCATED_BODY_LENGTH = 500;

    @Schema(title = "API Key")
    @NotNull
    @PluginProperty(secret = true, group = "main")
    private Property<String> apiKey;

    @Schema(
        title = "Model family",
        description = "Which Langdock Completion API route to list models from: `OPENAI` (default) or `ANTHROPIC`."
    )
    @Builder.Default
    @PluginProperty(group = "main")
    private Property<LangdockModelFamily> modelFamily = Property.ofValue(LangdockModelFamily.OPENAI);

    @Schema(
        title = "Region",
        description = "The Langdock region to list models from. Ignored when a dedicated-deployment `baseUrl` is set."
    )
    @Builder.Default
    @PluginProperty(group = "connection")
    private Property<LangdockRegion> region = Property.ofValue(LangdockRegion.EU);

    @Schema(
        title = "Base URL",
        description = "Custom base URL to override the default endpoint (useful for local tests, WireMock, or dedicated deployments)."
    )
    @PluginProperty(group = "connection")
    private Property<String> baseUrl;

    @Override
    public Output run(RunContext runContext) throws Exception {
        String rApiKey = runContext.render(this.apiKey).as(String.class).orElseThrow();
        LangdockModelFamily rFamily = runContext.render(this.modelFamily).as(LangdockModelFamily.class).orElse(LangdockModelFamily.OPENAI);
        LangdockRegion rRegion = runContext.render(this.region).as(LangdockRegion.class).orElse(LangdockRegion.EU);
        String rBaseUrl = runContext.render(this.baseUrl).as(String.class).orElse(null);
        String resolvedBaseUrl = rBaseUrl != null
            ? LangdockRoutes.normalizeBaseUrl(rBaseUrl, rFamily)
            : LangdockRoutes.chatBaseUrl(rRegion, rFamily);

        try (HttpClient httpClient = HttpClient.builder()
            .runContext(runContext)
            .configuration(HttpConfiguration.builder()
                .timeout(TimeoutConfiguration.builder()
                    .connectTimeout(Property.ofValue(Duration.ofSeconds(10)))
                    .readIdleTimeout(Property.ofValue(Duration.ofSeconds(30)))
                    .build())
                .build())
            .build()) {

            HttpRequest request = HttpRequest.builder()
                .method("GET")
                .uri(buildModelsUri(resolvedBaseUrl))
                .addHeader("Authorization", "Bearer " + rApiKey)
                .addHeader("Accept", "application/json")
                .build();

            HttpResponse<String> response;
            try {
                response = httpClient.request(request, String.class);
            } catch (HttpClientResponseException e) {
                throw mapError(e, rFamily);
            }

            List<Model> models = parseModels(response.getBody(), rFamily);
            runContext.logger().info("Retrieved {} Langdock model(s) from the {} route ({} region)", models.size(), rFamily, rRegion);

            return Output.builder()
                .models(models)
                .count(models.size())
                .build();
        }
    }

    private static URI buildModelsUri(String baseUrl) {
        return URI.create(baseUrl.endsWith("/") ? baseUrl + "models" : baseUrl + "/models");
    }

    @SuppressWarnings("unchecked")
    private List<Model> parseModels(String body, LangdockModelFamily family) {
        if (body == null || body.isBlank()) {
            throw new IllegalStateException("Empty response from Langdock while listing " + family + " models.");
        }

        Map<String, Object> envelope;
        try {
            envelope = MAPPER.readValue(body, new TypeReference<Map<String, Object>>() {
            });
        } catch (Exception e) {
            throw new IllegalStateException(
                "Unexpected Langdock response while listing " + family + " models — could not parse JSON. Response started with: " + truncate(body), e
            );
        }

        if (!(envelope.get("data") instanceof List<?> data)) {
            throw new IllegalStateException(
                "Unexpected Langdock response while listing " + family + " models — missing a 'data' array. Response started with: " + truncate(body)
            );
        }

        List<Model> models = new ArrayList<>();
        for (Object entry : data) {
            if (!(entry instanceof Map<?, ?> map)) {
                continue;
            }
            Map<String, Object> model = (Map<String, Object>) map;
            Object id = model.get("id");
            if (id == null) {
                continue;
            }
            models.add(Model.builder()
                .id(String.valueOf(id))
                .displayName(asString(model.get("display_name")))
                .ownedBy(asString(model.get("owned_by")))
                .createdAt(parseCreatedAt(model.get("created"), model.get("created_at")))
                .build());
        }
        return models;
    }

    private static String asString(Object value) {
        return value == null ? null : String.valueOf(value);
    }

    private Instant parseCreatedAt(Object createdEpochSeconds, Object createdAtIso) {
        if (createdEpochSeconds instanceof Number number) {
            return Instant.ofEpochSecond(number.longValue());
        }
        if (createdAtIso instanceof String iso) {
            try {
                return Instant.parse(iso);
            } catch (DateTimeParseException e) {
                return null;
            }
        }
        return null;
    }

    private String truncate(String body) {
        return body.length() > TRUNCATED_BODY_LENGTH ? body.substring(0, TRUNCATED_BODY_LENGTH) + "..." : body;
    }

    private IllegalStateException mapError(HttpClientResponseException e, LangdockModelFamily family) {
        int status = e.getResponse().getStatus().getCode();
        String body = rawBody(e);

        String hint = switch (status) {
            case 401, 403 ->
                " Check that 'apiKey' is a valid Langdock API key with the Completion API scope.";
            case 429 -> " Langdock rate limit exceeded — reduce call frequency or use Kestra retry.";
            default -> "";
        };

        return new IllegalStateException("Failed to list Langdock " + family + " models: HTTP " + status + " - " + truncate(body) + hint, e);
    }

    private String rawBody(HttpClientResponseException e) {
        Object body = e.getResponse().getBody();
        return switch (body) {
            case null -> "";
            case byte[] bytes -> new String(bytes, StandardCharsets.UTF_8);
            default -> String.valueOf(body);
        };
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Models", description = "The models available on the selected Langdock route.")
        private List<Model> models;

        @Schema(title = "Count", description = "The number of models returned.")
        private Integer count;
    }

    @Builder
    @Getter
    public static class Model {
        @Schema(title = "Model ID", description = "The model identifier to use as `modelName` on the Langdock provider.")
        private String id;

        @Schema(title = "Display name", description = "Human-readable model name, when provided by the Anthropic route.")
        private String displayName;

        @Schema(title = "Created at", description = "Model creation timestamp, when provided by the API.")
        private Instant createdAt;

        @Schema(title = "Owned by", description = "The organization that owns the model, when provided by the OpenAI route.")
        private String ownedBy;
    }
}
