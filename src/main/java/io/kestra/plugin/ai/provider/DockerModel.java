package io.kestra.plugin.ai.provider;

import java.net.InetAddress;
import java.net.URI;
import java.net.UnknownHostException;
import java.time.Duration;
import java.util.List;
import java.util.regex.Pattern;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.ai.domain.ChatConfiguration;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.openai.OpenAiImageModel;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(
    title = "Use Docker Model Runner",
    description = """
        Routes inference to a locally running Docker Model Runner instance via its OpenAI-compatible REST API.

        Docker Model Runner is built into Docker Desktop and Docker Engine (Linux) and requires no separate setup.
        It exposes an OpenAI-compatible API and does not require authentication — set `apiKey` to any non-empty \
        value (the default `not-needed` works).

        **Base URL variants** — pick the one matching where Kestra itself runs:
        - Kestra in a container on Docker Desktop: `http://model-runner.docker.internal/engines/v1`
        - Kestra in a container on Docker Engine (Linux): `http://172.17.0.1:12434/engines/v1`
        - Kestra directly on the host (default): `http://localhost:12434/engines/v1`

        The default suits a host installation. Most deployments run Kestra in a bridge-networked container, \
        where `localhost` is the Kestra container itself rather than the Docker Model Runner host — set \
        `baseUrl` explicitly in that case.

        Image generation routes to the Diffusers endpoint (`/engines/diffusers/v1`) automatically; use a \
        diffuser-capable model such as `ai/stable-diffusion`. Docker Model Runner does not advertise which \
        models are diffuser-capable and does not reject a chat model, so passing one makes the request hang \
        until it times out. The first image generation also downloads the Diffusers backend, which can take \
        several minutes.

        Pair this provider with `io.kestra.plugin.docker.model.Pull` (plugin-docker) to manage model lifecycle \
        in the same flow.
        """
)
@Plugin(
    examples = {
        @Example(
            title = "Chat completion with Docker Model Runner",
            full = true,
            code = {
                """
                    id: docker_model_chat
                    namespace: company.ai

                    inputs:
                      - id: prompt
                        type: STRING

                    tasks:
                      - id: pull_model
                        type: io.kestra.plugin.docker.model.Pull
                        model: ai/smollm2

                      - id: ask
                        type: io.kestra.plugin.ai.completion.ChatCompletion
                        provider:
                          type: io.kestra.plugin.ai.provider.DockerModel
                          modelName: ai/smollm2
                        messages:
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            }
        ),
        @Example(
            title = "Chat completion (container-internal base URL)",
            full = true,
            code = {
                """
                    id: docker_model_chat_container
                    namespace: company.ai

                    inputs:
                      - id: prompt
                        type: STRING

                    tasks:
                      - id: ask
                        type: io.kestra.plugin.ai.completion.ChatCompletion
                        provider:
                          type: io.kestra.plugin.ai.provider.DockerModel
                          modelName: ai/smollm2
                          baseUrl: http://model-runner.docker.internal/engines/v1
                        messages:
                          - type: USER
                            content: "{{ inputs.prompt }}"
                    """
            }
        )
    }
)
public class DockerModel extends OpenAICompliantProvider {

    static final String DEFAULT_BASE_URL = "http://localhost:12434/engines/v1";
    static final String DESKTOP_BASE_URL = "http://model-runner.docker.internal/engines/v1";
    static final String ENGINE_BASE_URL = "http://172.17.0.1:12434/engines/v1";
    // Docker Model Runner does no auth, but the OpenAI client requires a non-empty key, so send a placeholder.
    static final String PLACEHOLDER_API_KEY = "not-needed";
    private static final String DIFFUSER_PATH = "/engines/diffusers/v1";
    private static final String BASE_PATH = "/engines/v1";
    // Match /engines/v1 only as a whole path segment (end of URL or followed by '/'), so a longer
    // segment like /engines/v10 does not false-match and get silently rewritten.
    private static final Pattern BASE_PATH_SEGMENT = Pattern.compile(Pattern.quote(BASE_PATH) + "(?=/|$)");

    @Schema(
        title = "API base URL",
        description = """
            Base URL for the Docker Model Runner OpenAI-compatible API. Pick the variant matching where \
            Kestra itself runs:
            - Kestra in a container on Docker Desktop: `http://model-runner.docker.internal/engines/v1`
            - Kestra in a container on Docker Engine (Linux): `http://172.17.0.1:12434/engines/v1`
            - Kestra directly on the host (default): `http://localhost:12434/engines/v1`

            The `model-runner.docker.internal` alias exists only inside containers on Docker Desktop.
            """
    )
    @Builder.Default
    @PluginProperty(group = "connection")
    private Property<String> baseUrl = Property.ofValue(DEFAULT_BASE_URL);

    /**
     * Re-declared to give Docker Model Runner a working default, since {@code @Builder.Default} cannot be
     * applied to the inherited field. This shadows {@link OpenAICompliantProvider#apiKey}, which is why the
     * parent must read the API key through {@link #getApiKey()} rather than the field — the parent's own slot
     * stays null on instances of this class.
     */
    @Schema(
        title = "API Key",
        description = "Docker Model Runner does not require authentication. Any non-empty value is accepted; defaults to `not-needed`."
    )
    @Builder.Default
    @PluginProperty(secret = true, group = "main")
    private Property<String> apiKey = Property.ofValue(PLACEHOLDER_API_KEY);

    @Override
    public ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout, List<ChatModelListener> additionalListeners)
        throws IllegalVariableEvaluationException {
        assertHostResolvable(runContext);
        return super.chatModel(runContext, configuration, timeout, additionalListeners);
    }

    @Override
    public EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        assertHostResolvable(runContext);
        return super.embeddingModel(runContext);
    }

    /**
     * Routes image generation to the Diffusers endpoint by replacing the chat path segment.
     * Requires a diffuser-capable model (e.g. ai/stable-diffusion).
     */
    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        String resolvedBaseUrl = runContext.render(getBaseUrl()).as(String.class).orElse(DEFAULT_BASE_URL);
        var matcher = BASE_PATH_SEGMENT.matcher(resolvedBaseUrl);
        if (!matcher.find()) {
            throw new IllegalArgumentException(
                "Cannot derive the Docker Model Runner Diffusers endpoint from baseUrl '" + resolvedBaseUrl +
                    "': expected it to contain the path segment '" + BASE_PATH + "'. Set baseUrl to a Docker Model " +
                    "Runner OpenAI-compatible endpoint, e.g. http://localhost:12434/engines/v1."
            );
        }
        assertHostResolvable(resolvedBaseUrl);
        String diffuserUrl = matcher.replaceFirst(DIFFUSER_PATH);
        return OpenAiImageModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .apiKey(resolveApiKey(runContext))
            .baseUrl(diffuserUrl)
            .build();
    }

    private void assertHostResolvable(RunContext runContext) throws IllegalVariableEvaluationException {
        assertHostResolvable(runContext.render(getBaseUrl()).as(String.class).orElse(DEFAULT_BASE_URL));
    }

    /**
     * Fails fast when the Docker Model Runner host cannot be resolved.
     * <p>
     * LangChain4j only contacts the server at inference time, long after this provider has returned, and
     * surfaces an unresolvable host as {@code UnresolvedModelServerException} with a {@code null} message for
     * chat and embeddings, or a bare {@code ConnectException} for images. Neither names the host, so a user who
     * points {@code baseUrl} at the Docker Desktop alias while running on Docker Engine gets no clue what went
     * wrong. Resolving the host here converts that into an actionable error at configuration time.
     * <p>
     * Only name resolution is pre-empted. Every other failure — connection refused, timeouts, HTTP errors —
     * still propagates untouched from the model itself.
     */
    private void assertHostResolvable(String resolvedBaseUrl) {
        String host;
        try {
            host = URI.create(resolvedBaseUrl).getHost();
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                "Invalid Docker Model Runner baseUrl '" + resolvedBaseUrl + "': " + e.getMessage(), e
            );
        }

        if (host == null) {
            throw new IllegalArgumentException(
                "Invalid Docker Model Runner baseUrl '" + resolvedBaseUrl + "': it has no host component. " +
                    "Expected an absolute URL such as " + DEFAULT_BASE_URL + "."
            );
        }

        try {
            InetAddress.getByName(host);
        } catch (UnknownHostException e) {
            throw new IllegalArgumentException(
                "Cannot resolve host '" + host + "' from Docker Model Runner baseUrl '" + resolvedBaseUrl +
                    "'. Set baseUrl to match where Kestra runs:" +
                    "\n  - Kestra in a container on Docker Desktop: " + DESKTOP_BASE_URL +
                    "\n  - Kestra in a container on Docker Engine (Linux): " + ENGINE_BASE_URL +
                    "\n  - Kestra directly on the host: " + DEFAULT_BASE_URL +
                    "\nNote that the model-runner.docker.internal alias exists only inside containers on " +
                    "Docker Desktop.",
                e
            );
        }
    }
}
