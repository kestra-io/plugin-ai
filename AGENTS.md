# Kestra AI Plugin

## What

Plugin for AI with Langchain4j for Kestra Exposes 57 plugin components (tasks, triggers, and/or conditions).

## Why

Enables Kestra workflows to interact with AI, allowing orchestration of AI-based operations as part of data pipelines and automation workflows.

## How

### Architecture

Single-module plugin. Source packages under `io.kestra.plugin`:

- `ai`

Infrastructure dependencies (Docker Compose services):

- `app`

### Key Plugin Classes

- `io.kestra.plugin.ai.agent.A2AClient`
- `io.kestra.plugin.ai.agent.AIAgent`
- `io.kestra.plugin.ai.completion.ChatCompletion`
- `io.kestra.plugin.ai.completion.Classification`
- `io.kestra.plugin.ai.completion.ImageGeneration`
- `io.kestra.plugin.ai.completion.JSONStructuredExtraction`
- `io.kestra.plugin.ai.embeddings.Chroma`
- `io.kestra.plugin.ai.embeddings.Elasticsearch`
- `io.kestra.plugin.ai.embeddings.KestraKVStore`
- `io.kestra.plugin.ai.embeddings.MariaDB`
- `io.kestra.plugin.ai.embeddings.Milvus`
- `io.kestra.plugin.ai.embeddings.MongoDBAtlas`
- `io.kestra.plugin.ai.embeddings.PGVector`
- `io.kestra.plugin.ai.embeddings.Pinecone`
- `io.kestra.plugin.ai.embeddings.Qdrant`
- `io.kestra.plugin.ai.embeddings.Redis`
- `io.kestra.plugin.ai.embeddings.Tablestore`
- `io.kestra.plugin.ai.embeddings.Weaviate`
- `io.kestra.plugin.ai.memory.KestraKVStore`
- `io.kestra.plugin.ai.memory.PostgreSQL`
- `io.kestra.plugin.ai.memory.Redis`
- `io.kestra.plugin.ai.provider.AmazonBedrock`
- `io.kestra.plugin.ai.provider.Anthropic`
- `io.kestra.plugin.ai.provider.AzureOpenAI`
- `io.kestra.plugin.ai.provider.DashScope`
- `io.kestra.plugin.ai.provider.DeepSeek`
- `io.kestra.plugin.ai.provider.GitHubModels`
- `io.kestra.plugin.ai.provider.GoogleGemini`
- `io.kestra.plugin.ai.provider.GoogleVertexAI`
- `io.kestra.plugin.ai.provider.HuggingFace`
- `io.kestra.plugin.ai.provider.LocalAI`
- `io.kestra.plugin.ai.provider.MistralAI`
- `io.kestra.plugin.ai.provider.OciGenAI`
- `io.kestra.plugin.ai.provider.Ollama`
- `io.kestra.plugin.ai.provider.OpenAI`
- `io.kestra.plugin.ai.provider.OpenRouter`
- `io.kestra.plugin.ai.provider.WatsonxAI`
- `io.kestra.plugin.ai.provider.WorkersAI`
- `io.kestra.plugin.ai.provider.ZhiPuAI`
- `io.kestra.plugin.ai.rag.ChatCompletion`
- `io.kestra.plugin.ai.rag.IngestDocument`
- `io.kestra.plugin.ai.rag.Search`
- `io.kestra.plugin.ai.retriever.EmbeddingStoreRetriever`
- `io.kestra.plugin.ai.retriever.GoogleCustomWebSearch`
- `io.kestra.plugin.ai.retriever.SqlDatabaseRetriever`
- `io.kestra.plugin.ai.retriever.TavilyWebSearch`
- `io.kestra.plugin.ai.tool.A2AClient`
- `io.kestra.plugin.ai.tool.AIAgent`
- `io.kestra.plugin.ai.tool.CodeExecution`
- `io.kestra.plugin.ai.tool.DockerMcpClient`
- `io.kestra.plugin.ai.tool.GoogleCustomWebSearch`
- `io.kestra.plugin.ai.tool.KestraFlow`
- `io.kestra.plugin.ai.tool.KestraTask`
- `io.kestra.plugin.ai.tool.SseMcpClient`
- `io.kestra.plugin.ai.tool.StdioMcpClient`
- `io.kestra.plugin.ai.tool.StreamableHttpMcpClient`
- `io.kestra.plugin.ai.tool.TavilyWebSearch`

### Project Structure

```
plugin-ai/
├── src/main/java/io/kestra/plugin/ai/tool/
├── src/test/java/io/kestra/plugin/ai/tool/
├── build.gradle
└── README.md
```

### Important Commands

```bash
# Build the plugin
./gradlew shadowJar

# Run tests
./gradlew test

# Build without tests
./gradlew shadowJar -x test
```

### Configuration

All tasks and triggers accept standard Kestra plugin properties. Credentials should use
`{{ secret('SECRET_NAME') }}` — never hardcode real values.

## Agents

**IMPORTANT:** This is a Kestra plugin repository (prefixed by `plugin-`, `storage-`, or `secret-`). You **MUST** delegate all coding tasks to the `kestra-plugin-developer` agent. Do NOT implement code changes directly — always use this agent.
