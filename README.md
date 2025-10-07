<p align="center">
  <a href="https://www.kestra.io">
    <img src="https://kestra.io/banner.png" alt="Kestra workflow orchestrator" />
  </a>
</p>

<h1 align="center" style="border-bottom: none">
  Kestra AI Plugin
</h1>

<div align="center">
  <a href="https://github.com/kestra-io/kestra/releases"><img src="https://img.shields.io/github/tag-pre/kestra-io/kestra.svg?color=blueviolet" alt="Last Version" /></a>
  <a href="https://github.com/kestra-io/kestra/blob/develop/LICENSE"><img src="https://img.shields.io/github/license/kestra-io/kestra?color=blueviolet" alt="License" /></a>
  <a href="https://github.com/kestra-io/kestra/stargazers"><img src="https://img.shields.io/github/stars/kestra-io/kestra?color=blueviolet&logo=github" alt="Github star" /></a><br>
  <a href="https://kestra.io"><img src="https://img.shields.io/badge/Website-kestra.io-192A4E?color=blueviolet" alt="Kestra infinitely scalable orchestration and scheduling platform"></a>
  <a href="https://kestra.io/slack"><img src="https://img.shields.io/badge/Slack-Join%20Community-blueviolet?logo=slack" alt="Slack"></a>
</div>

<br/>

<p align="center">
  <a href="https://twitter.com/kestra_io"><img height="25" src="https://kestra.io/twitter.svg" alt="twitter" width="35"/></a>
  <a href="https://www.linkedin.com/company/kestra/"><img height="25" src="https://kestra.io/linkedin.svg" alt="linkedin" width="35"/></a>
  <a href="https://www.youtube.com/@kestra-io"><img height="25" src="https://kestra.io/youtube.svg" alt="youtube" width="35"/></a>
</p>

---

## 🧠 Overview

**Kestra AI Plugin** brings AI-native orchestration to [Kestra](https://github.com/kestra-io/kestra).
It enables workflows to leverage large language models (LLMs), vector databases, AI providers, and intelligent agents — declaratively.

From prompt-based completions to full Retrieval-Augmented Generation (RAG) pipelines, this plugin provides all the tools you need to build **AI-driven automations** within Kestra.

---

## ✨ Features

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

---

## ⚙️ Example Use Cases

- Build **chatbots** that query your data using RAG pipelines.
- Use **LLMs to classify**, **summarize**, or **extract** information from text.
- Automate **image generation** tasks.
- Integrate AI directly in **ETL or data processing pipelines**.
- Orchestrate complex **multi-agent reasoning workflows** using `AIAgent` and `KestraFlow`.
