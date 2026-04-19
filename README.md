<p align="center">
  <a href="https://www.kestra.io">
    <img src="https://kestra.io/banner.png"  alt="Kestra workflow orchestrator" />
  </a>
</p>

<h1 align="center" style="border-bottom: none">
    Event-Driven Declarative Orchestrator
</h1>

<div align="center">
 <a href="https://github.com/kestra-io/kestra/releases"><img src="https://img.shields.io/github/tag-pre/kestra-io/kestra.svg?color=blueviolet" alt="Last Version" /></a>
  <a href="https://github.com/kestra-io/kestra/blob/develop/LICENSE"><img src="https://img.shields.io/github/license/kestra-io/kestra?color=blueviolet" alt="License" /></a>
  <a href="https://github.com/kestra-io/kestra/stargazers"><img src="https://img.shields.io/github/stars/kestra-io/kestra?color=blueviolet&logo=github" alt="Github star" /></a> <br>
<a href="https://kestra.io"><img src="https://img.shields.io/badge/Website-kestra.io-192A4E?color=blueviolet" alt="Kestra infinitely scalable orchestration and scheduling platform"></a>
<a href="https://kestra.io/slack"><img src="https://img.shields.io/badge/Slack-Join%20Community-blueviolet?logo=slack" alt="Slack"></a>
</div>

<br />

<p align="center">
  <a href="https://twitter.com/kestra_io" style="margin: 0 10px;">
        <img src="https://kestra.io/twitter.svg" alt="twitter" width="35" height="25" /></a>
  <a href="https://www.linkedin.com/company/kestra/" style="margin: 0 10px;">
        <img src="https://kestra.io/linkedin.svg" alt="linkedin" width="35" height="25" /></a>
  <a href="https://www.youtube.com/@kestra-io" style="margin: 0 10px;">
        <img src="https://kestra.io/youtube.svg" alt="youtube" width="35" height="25" /></a>
</p>

<br />
<p align="center">
    <a href="https://go.kestra.io/video/product-overview" target="_blank">
        <img src="https://kestra.io/startvideo.png" alt="Get started in 4 minutes with Kestra" width="640px" />
    </a>
</p>
<p align="center" style="color:grey;"><i>Get started in 4 minutes with Kestra.</i></p>

# Kestra Ai Plugin

## Why

- What user problem does this solve? Teams need to orchestrate generative AI in Kestra with LangChain4j, covering chat completions, agents, RAG, tools, and shared providers from orchestrated workflows instead of relying on manual console work, ad hoc scripts, or disconnected schedulers.
- Why would a team adopt this plugin in a workflow? It keeps AI steps in the same Kestra flow as upstream preparation, approvals, retries, notifications, and downstream systems.
- What operational/business outcome does it enable? It reduces manual handoffs and fragmented tooling while improving reliability, traceability, and delivery speed for processes that depend on AI.

## What

- Provides plugin components under `io.kestra.plugin.ai`.
- Includes classes such as `AIUtils`, `GoogleCustomWebSearch`, `TavilyWebSearch`, `SqlDatabaseRetriever`.

### Features

### 🗣️ Chat & Completion
- `ChatCompletion` — generic chat interface for LLMs (OpenAI, Anthropic, Gemini, etc.)
- `Classification` — classify text into predefined categories
- `ImageGeneration` — create images from text prompts
- `JSONStructuredExtraction` — extract structured data (JSON) from unstructured text

### 🧩 Providers
Supports multiple AI backends:
- **OpenAI**, **Azure OpenAI**, **Anthropic**, **MistralAI**, **Google VertexAI**, **Google Gemini**
- **Amazon Bedrock**, **Ollama**, **LocalAI**, **OpenRouter**, **DeepSeek**, **DashScope**
- **WorkersAI** (Cloudflare)

Each provider integrates authentication, model selection, and latency tracking through `TimingChatModelListener`.

### 🧱 Embeddings
Store and search semantic vector embeddings using:
- **PGVector**
- **Elasticsearch**
- **Milvus**
- **Weaviate**
- **Chroma**
- **Pinecone**
- **Qdrant**
- **Redis**
- **KestraKVStore**
- **MongoDB Atlas**

These integrations make it easy to build **semantic search**, **context retrieval**, or **memory-augmented workflows**.

### 🧠 Memory
Persistent conversational or vector-based memory via:
- `Redis`
- `KestraKVStore`

### 🔍 Retrieval & RAG
Tools for **Retrieval-Augmented Generation (RAG)**:
- `IngestDocument` — chunk and embed documents
- `Search` — perform vector similarity searches
- `ChatCompletion` — combine retrieved context with a chat model

### 🌐 Tools & Agents
The plugin includes a flexible **Agent framework** and **tools** that can interact with external systems:
- `AIAgent` — orchestrates tool usage dynamically
- `CodeExecution` — safely run code blocks
- `KestraTask` & `KestraFlow` — trigger Kestra workflows and tasks from within an agent
- `GoogleCustomWebSearch`, `TavilyWebSearch` — retrieve web content
- MCP Clients (`DockerMcpClient`, `SseMcpClient`, `StdioMcpClient`, `StreamableHttpMcpClient`) — connect to external Model Context Protocol tools

### Example Use Cases

- Build **chatbots** that query your data using RAG pipelines.
- Use **LLMs to classify**, **summarize**, or **extract** information from text.
- Automate **image generation** tasks.
- Integrate AI directly in **ETL or data processing pipelines**.
- Orchestrate complex **multi-agent reasoning workflows** using `AIAgent` and `KestraFlow`.

## Documentation
* Full documentation can be found under: [kestra.io/docs](https://kestra.io/docs)
* Documentation for developing a plugin is included in the [Plugin Developer Guide](https://kestra.io/docs/plugin-developer-guide/)


## License
Apache 2.0 © [Kestra Technologies](https://kestra.io)

## Stay up to date

We release new versions every month. Give the [main repository](https://github.com/kestra-io/kestra) a star to stay up to date with the latest releases and get notified about future updates.

![Star the repo](https://kestra.io/star.gif)
