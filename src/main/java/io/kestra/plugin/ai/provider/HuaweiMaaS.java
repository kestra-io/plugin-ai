package io.kestra.plugin.ai.provider;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Getter
@SuperBuilder
@NoArgsConstructor
@JsonDeserialize
@Schema(title = "Use Huawei Cloud MaaS models", description = "Connects to Huawei Cloud MaaS using its OpenAI-compatible chat API. The documented endpoint is currently available in the CN-Hong Kong region.")
@Plugin(examples = {
        @Example(title = "Chat completion with Huawei Cloud MaaS", full = true, code = {
                """
                        id: huawei_maas_chat
                        namespace: company.ai

                        inputs:
                          - id: prompt
                            type: STRING

                        tasks:
                          - id: chat
                            type: io.kestra.plugin.ai.completion.ChatCompletion
                            provider:
                              type: io.kestra.plugin.ai.provider.HuaweiMaaS
                              apiKey: "{{ secret('HUAWEI_MAAS_API_KEY') }}"
                              modelName: glm-5.2
                            messages:
                              - type: SYSTEM
                                content: You are a helpful assistant. Answer clearly and concisely.
                              - type: USER
                                content: "{{ inputs.prompt }}"
                        """
        }),
        @Example(title = "Extract structured data using Huawei MaaS", full = true, code = {
                """
                        id: huawei_maas_structured_extraction
                        namespace: company.ai

                        tasks:
                          - id: extract_person
                            type: io.kestra.plugin.ai.completion.JSONStructuredExtraction
                            schemaName: Person
                            jsonFields:
                              - name
                              - city
                              - country
                              - email
                            prompt: |
                              Extract the person's name, city, country, and email
                              from the following text:
                              "I'm John Smith from Paris, France.
                              Contact me at john.smith@example.com."
                            systemMessage: You extract structured data in JSON format.
                            provider:
                              type: io.kestra.plugin.ai.provider.HuaweiMaaS
                              apiKey: "{{ secret('HUAWEI_MAAS_API_KEY') }}"
                              modelName: glm-5.2
                        """
        }),

        @Example(title = "RAG chat with Huawei MaaS and Gemini embeddings", full = true, code = {
                """
                        id: huawei_maas_rag
                        namespace: company.ai

                        tasks:
                          - id: ingest
                            type: io.kestra.plugin.ai.rag.IngestDocument
                            provider:
                              type: io.kestra.plugin.ai.provider.GoogleGemini
                              modelName: gemini-embedding-001
                              apiKey: "{{ secret('GEMINI_API_KEY') }}"
                            embeddings:
                              type: io.kestra.plugin.ai.embeddings.KestraKVStore
                            drop: true
                            fromExternalURLs:
                              - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md

                          - id: chat_with_rag
                            type: io.kestra.plugin.ai.rag.ChatCompletion
                            chatProvider:
                              type: io.kestra.plugin.ai.provider.HuaweiMaaS
                              apiKey: "{{ secret('HUAWEI_MAAS_API_KEY') }}"
                              modelName: glm-5.2
                            embeddingProvider:
                              type: io.kestra.plugin.ai.provider.GoogleGemini
                              modelName: gemini-embedding-001
                              apiKey: "{{ secret('GEMINI_API_KEY') }}"
                            embeddings:
                              type: io.kestra.plugin.ai.embeddings.KestraKVStore
                            systemMessage: You are a helpful assistant that answers using the provided documents.
                            prompt: Which features were released in Kestra 0.24?
                        """
        })

})
public class HuaweiMaaS extends OpenAICompliantProvider {
    private static final String BASE_URL = "https://api-ap-southeast-1.modelarts-maas.com/openai/v1";

    @Schema(title = "API base URL", description = "Huawei Cloud MaaS OpenAI-compatible API base URL. The documented endpoint is currently available in the CN-Hong Kong region.", example = "https://api-ap-southeast-1.modelarts-maas.com/openai/v1")
    @Builder.Default
    @PluginProperty(group = "connection")
    private Property<String> baseUrl = Property.ofValue(BASE_URL);

    @Override
    public ImageModel imageModel(io.kestra.core.runners.RunContext runContext)
            throws IllegalVariableEvaluationException {
        throw new IllegalArgumentException(
                "Huawei Cloud MaaS provider does not currently support image generation through its documented OpenAI-compatible API.");
    }

    @Override
    public EmbeddingModel embeddingModel(io.kestra.core.runners.RunContext runContext)
            throws IllegalVariableEvaluationException {
        throw new IllegalArgumentException(
                "Huawei Cloud MaaS provider does not currently support embeddings through its documented OpenAI-compatible API.");
    }
}
