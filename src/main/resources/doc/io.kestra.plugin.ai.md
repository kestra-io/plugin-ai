# How to use the LangChain4J plugin

Run AI completions, structured extraction, image generation, RAG pipelines, and autonomous agents from Kestra flows.

## Authentication

All tasks require a `provider` object that selects the LLM backend and carries its credentials. Set `modelName` on each provider. Available providers: `OpenAI`, `Anthropic`, `GoogleGemini`, `GoogleVertexAI`, `AzureOpenAI`, `Ollama`, `MistralAI`, `AmazonBedrock`, `GitHubModels`, `HuggingFace`, `LocalAI`, `OciGenAI`, `OpenRouter`, `DeepSeek`, `DashScope`, `WatsonxAI`, `WorkersAI`, `ZhiPuAI`, and `OpenAICompliantProvider` for any OpenAI-compatible endpoint.

Most providers set `apiKey` (required for their respective service). `AzureOpenAI` uses `endpoint` plus `apiKey` or OAuth credentials. `Ollama` uses `endpoint` (e.g. `http://localhost:11434`). `GoogleVertexAI` uses service account credentials.

Store secrets in [secrets](https://kestra.io/docs/concepts/secret) and apply connection properties globally with [plugin defaults](https://kestra.io/docs/workflow-components/plugin-defaults).

## Common properties

All tasks that generate text accept a `configuration` object (`ChatConfiguration`) to tune the model: set `temperature`, `topK`, `topP`, `seed`, `maxToken`, `logRequests`, `logResponses`. Enable extended thinking with `thinkingEnabled`, `thinkingBudgetTokens`, and `returnThinking`. Control response format with `configuration.responseFormat` — set `type` to `TEXT` (default) or `JSON_SCHEMA`; for JSON output also set `jsonSchema` and optionally `strictJson`.

## Tasks

### Completions

`completion.ChatCompletion` sends a multi-turn conversation to the model — set `messages` (required, list of objects with `type` (`SYSTEM`, `USER`, or `AI`) and `content`) and `provider`. Optionally pass `configuration`, `tools` (list of tool providers), and `memory`. The output includes `textOutput`, `jsonOutput`, `tokenUsage`, `finishReason`, `toolExecutions`, `thinking`, and `sources`.

`completion.Classification` classifies text — set `prompt` (required), `classes` (required, list of label strings), and `provider`. Optionally override `systemMessage` and pass `configuration`. The output includes `classification`, `tokenUsage`, and `finishReason`.

`completion.ImageGeneration` generates an image from a text description — set `prompt` (required) and `provider`. The output includes `imageUrl`, `tokenUsage`, and `finishReason`.

`completion.JSONStructuredExtraction` extracts structured data from text — set `schemaName` (required), `jsonFields` (required, list of field names to extract), and `provider`. Optionally set `prompt`, `systemMessage`, and `configuration`. The output includes `schemaName`, `extractedJson`, `tokenUsage`, and `finishReason`.

### RAG

`rag.IngestDocument` ingests documents into an embedding store — set `provider` (embedding model, required) and `embeddings` (store, required). Provide content via one or more of: `fromPath` (path in task working directory), `fromInternalURIs` (list of `kestra://` URIs), `fromExternalURLs` (list of HTTP URLs), or `fromDocuments` (inline list with `content` and optional `metadata`). Control chunking with `documentSplitter` (set `splitter` type: `RECURSIVE`, `PARAGRAPH`, `LINE`, `SENTENCE`, or `WORD`; plus `maxSegmentSizeInChars` and `maxOverlapSizeInChars`). Set `drop: true` to clear the store before ingesting. Set `bulkSize` to control batch size (default 500). The output includes `ingestedDocuments`, `inputTokenCount`, `outputTokenCount`, and `totalTokenCount`.

`rag.Search` retrieves matching chunks from an embedding store — set `query` (required), `maxResults` (required), `minScore` (required), `provider` (required), and `embeddings` (required). Set `fetchType` to control output (default `NONE`; use `FETCH` or `STORE` to surface results). The output includes `results`, `uri`, and `size`.

`rag.ChatCompletion` runs a retrieval-augmented chat — set `prompt` (required) and `chatProvider` (required, the LLM). Configure retrieval via `embeddings` and `embeddingProvider`, or provide `contentRetrievers` (a list of `EmbeddingStoreRetriever` or `TavilyWebSearch` retriever objects). Control retrieval with `contentRetrieverConfiguration` (defaults: `maxResults` 3, `minScore` 0.0). Optionally set `systemMessage`, `tools`, `memory`, and `chatConfiguration`.

### Agent

`agent.AIAgent` runs an autonomous agent loop — set `prompt` (required) and `provider` (required). Attach callable tools via `tools` (list of tool providers). Optionally set `systemMessage`, `configuration`, `memory`, `contentRetrievers`, `maxSequentialToolsInvocations`, `outputFiles`, and `observability` (LangfuseObservability for tracing).

## Embedding stores

Configure `embeddings` with one of these subtypes:

- **KestraKVStore** — uses Kestra's built-in KV store; optionally set `kvName` (default `{{flow.id}}-embedding-store`)
- **Pinecone** — set `apiKey`, `cloud`, `region`, and `index` (all required); optionally `namespace`
- **Qdrant** — set `apiKey`, `host`, `port`, and `collectionName` (all required)
- **Redis** — set `host` and `port` (both required); optionally set `indexName` (default `embedding-index`)
- **PGVector** — set `host`, `port`, `user`, `password`, `database`, and `table` (all required); optionally `useIndex` (default false)
- **MongoDBAtlas** — set `apiKey`, `collectionName`, and `databaseName` (all required)
- **Chroma** — connect to a Chroma vector database
- **Weaviate** — connect to a Weaviate instance
- **MariaDB** — use MariaDB as a vector store
- **Milvus** — connect to a Milvus cluster
- **Elasticsearch** — connect to an Elasticsearch index
- **Tablestore** — connect to Alibaba Cloud Tablestore

## Memory providers

Configure `memory` on chat and agent tasks with one of these subtypes:

- **KestraKVStore** — persists memory in Kestra's KV store
- **Redis** — set `host` (required) and `port` (default 6379)
- **PostgreSQL** — set `host`, `database`, `user`, and `password` (all required); optionally `port` (default 5432) and `tableName` (default `chat_memory`)

All memory providers share `messages` (default 10), `ttl` (default PT1H), `memoryId` (default `{{labels.system.correlationId}}`), and `drop` (default `NEVER`; also `BEFORE_TASKRUN` or `AFTER_TASKRUN`).

## Tools

Configure `tools` on chat and agent tasks with one of these subtypes:

- **TavilyWebSearch** — set `apiKey` (required)
- **CodeExecution** — set `apiKey` (required, RapidAPI key for Judge0)
- **GoogleCustomWebSearch** — set `apiKey` and `csi` (Custom Search Engine ID, both required)
- **KestraFlow** / **KestraTask** — invoke other Kestra flows or tasks as tools
- **SseMcpClient** / **StdioMcpClient** / **StreamableHttpMcpClient** / **DockerMcpClient** — connect to MCP servers
- **A2AClient** — connect to an A2A-compatible agent
