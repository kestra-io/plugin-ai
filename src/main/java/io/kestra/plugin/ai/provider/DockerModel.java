package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;

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

        **Base URL variants:**
        - Host (default): `http://localhost:12434/engines/v1`
        - Docker Desktop container: `http://model-runner.docker.internal/engines/v1`
        - Docker Engine container: `http://172.17.0.1:12434/engines/v1`

        Image generation routes to the Diffusers endpoint (`/engines/diffusers/v1`) automatically; use a \
        diffuser-tagged model such as `ai/stable-diffusion`.

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
                          - role: user
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
                          - role: user
                            content: "{{ inputs.prompt }}"
                    """
            }
        )
    }
)
public class DockerModel extends OpenAICompliantProvider {

    static final String DEFAULT_BASE_URL = "http://localhost:12434/engines/v1";
    private static final String DIFFUSER_PATH = "/engines/diffusers/v1";
    private static final String CHAT_PATH = "/engines/v1";

    @Schema(
        title = "API base URL",
        description = """
            Base URL for the Docker Model Runner OpenAI-compatible API.
            - Host access (default): `http://localhost:12434/engines/v1`
            - Docker Desktop container: `http://model-runner.docker.internal/engines/v1`
            - Docker Engine container: `http://172.17.0.1:12434/engines/v1`
            """
    )
    @Builder.Default
    @PluginProperty(group = "connection")
    private Property<String> baseUrl = Property.ofValue(DEFAULT_BASE_URL);

    @Schema(
        title = "API Key",
        description = "Docker Model Runner does not require authentication. Any non-empty value is accepted; defaults to `not-needed`."
    )
    @Builder.Default
    @PluginProperty(secret = true, group = "main")
    private Property<String> apiKey = Property.ofValue("not-needed");

    @Override
    protected String resolveApiKey(RunContext runContext) throws IllegalVariableEvaluationException {
        return runContext.render(this.apiKey).as(String.class).orElse("not-needed");
    }

    /**
     * Routes image generation to the Diffusers endpoint by replacing the chat path segment.
     * Requires a diffuser-tagged model (e.g. ai/stable-diffusion).
     */
    @Override
    public ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        String resolvedBaseUrl = runContext.render(this.baseUrl).as(String.class).orElse(DEFAULT_BASE_URL);
        String diffuserUrl = resolvedBaseUrl.replace(CHAT_PATH, DIFFUSER_PATH);
        return OpenAiImageModel.builder()
            .modelName(runContext.render(this.getModelName()).as(String.class).orElseThrow())
            .apiKey(resolveApiKey(runContext))
            .baseUrl(diffuserUrl)
            .build();
    }
}
