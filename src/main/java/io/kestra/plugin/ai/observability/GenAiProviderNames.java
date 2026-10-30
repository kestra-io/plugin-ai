package io.kestra.plugin.ai.observability;

/**
 * Maps plugin provider classes to {@code gen_ai.provider.name}. Well-known semantic-convention values are used when
 * one applies; everything else is a stable custom value.
 */
final class GenAiProviderNames {
    private GenAiProviderNames() {
    }

    static String fromClass(Class<?> type) {
        var simpleName = type.getSimpleName();
        var known = switch (simpleName) {
            case "AmazonBedrock" -> "aws.bedrock";
            case "AzureOpenAI" -> "azure.ai.openai";
            case "DashScope" -> "dashscope";
            case "DeepSeek" -> "deepseek";
            case "DockerModel" -> "docker.model_runner";
            case "GitHubModels" -> "github.models";
            case "GoogleGemini" -> "gcp.gemini";
            case "GoogleVertexAI" -> "gcp.vertex_ai";
            case "HuggingFace" -> "huggingface";
            case "Langdock" -> "langdock";
            case "LocalAI" -> "localai";
            case "MistralAI" -> "mistral_ai";
            case "OciGenAI" -> "oracle.oci_genai";
            case "Ollama" -> "ollama";
            case "OpenRouter" -> "openrouter";
            case "WatsonxAI" -> "ibm.watsonx.ai";
            case "WorkersAI" -> "cloudflare.workers_ai";
            case "ZhiPuAI" -> "zhipuai";
            default -> null;
        };
        if (known != null) {
            return known;
        }
        if (simpleName.contains("Anthropic")) {
            return "anthropic";
        }
        if (simpleName.contains("OpenAI") || simpleName.contains("OpenAi")) {
            return "openai";
        }
        return simpleName.isBlank() ? "unknown" : simpleName;
    }
}
