package io.kestra.plugin.ai.agent;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.storages.StorageInterface;
import io.kestra.core.tenant.TenantService;
import io.kestra.core.utils.IdUtils;
import io.kestra.plugin.ai.domain.ChatConfiguration;
import io.kestra.plugin.ai.memory.KestraKVStore;
import io.kestra.plugin.ai.provider.GoogleGemini;
import io.kestra.plugin.ai.provider.OpenAI;
import io.kestra.plugin.ai.rag.IngestDocument;
import io.kestra.plugin.ai.retriever.EmbeddingStoreRetriever;
import io.kestra.plugin.ai.retriever.GoogleCustomWebSearch;
import io.kestra.plugin.ai.retriever.TavilyWebSearch;
import io.kestra.plugin.ai.tool.DockerMcpClient;
import io.kestra.plugin.ai.tool.StdioMcpClient;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class AIAgentTest {
    private static final String PINECONE_API_KEY = System.getenv("PINECONE_API_KEY");
    private final String GOOGLE_API_KEY = System.getenv("GOOGLE_API_KEY");
    private final String GOOGLE_CSI = System.getenv("GOOGLE_CSI_KEY");
    private final String TAVILY_API_KEY = System.getenv("TAVILY_API_KEY");
    @Inject
    private TestRunContextFactory runContextFactory;

    @Inject
    private StorageInterface storage;

    @Test
    void prompt() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1"
        ));

        var agent = AIAgent.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .systemMessage(Property.ofValue("You are a summary agent, summarize the test from the user message."))
            .prompt(Property.ofValue("Each flow can produce outputs that can be consumed by other flows. This is a list property, so that your flow can produce as many outputs as you need. Each output needs to have an id (the name of the output), a type (the same types you know from inputs e.g. STRING, URI or JSON) and value which is the actual output value that will be stored in internal storage and passed to other flows when needed."))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = agent.run(runContext);
        assertThat(output.getTextOutput()).isNotNull();
    }

    @Test
    void withTool() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1"
        ));

        var agent = AIAgent.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .tools(
                List.of(StdioMcpClient.builder().command(Property.ofValue(List.of("docker", "run", "--rm", "-i", "mcp/everything"))).build())
            )
            .prompt(Property.ofValue("What is 5+12? Use the provided tool to answer and always assume that the tool is correct."))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = agent.run(runContext);
        assertThat(output.getTextOutput()).isNotNull();
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("add");
    }

    @Test
    void withMemory() throws Exception {
        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1",
            "labels", Map.of("system", Map.of("correlationId", IdUtils.create()))
        ));

        var agent = AIAgent.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .prompt(Property.ofValue("My name is John."))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .memory(KestraKVStore.builder().build())
            .build();
        var output = agent.run(runContext);
        assertThat(output.getTextOutput()).isNotNull();

        agent = AIAgent.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .prompt(Property.ofValue("What's my name."))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .memory(KestraKVStore.builder().build())
            .build();
        output = agent.run(runContext);
        assertThat(output.getTextOutput()).contains("John");
    }

    @Test
    void withOutputFiles() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1"
        ));

        var agent = AIAgent.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .tools(
                List.of(DockerMcpClient.builder()
                    .image(Property.ofValue("mcp/filesystem"))
                    .command(Property.ofValue(List.of("/tmp")))
                    .binds(Property.ofValue(List.of(runContext.workingDir().path(true).toString() + ":/tmp")))
                    .logEvents(Property.ofValue(true))
                    .build()
                )
            )
            .prompt(Property.ofValue("Create a file '/tmp/hello.txt' with the content \"Hello World\""))
            // Use a low temperature and a fixed seed so the completion would be more deterministic
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .outputFiles(Property.ofValue(List.of("hello.txt")))
            .build();

        var output = agent.run(runContext);
        assertThat(output.getTextOutput()).isNotNull();
        assertThat(output.getToolExecutions()).isNotEmpty();
        assertThat(output.getToolExecutions()).extracting("requestName").contains("write_file");
        assertThat(output.getOutputFiles()).hasSize(1);
        assertThat(output.getOutputFiles().get("hello.txt")).isNotNull();

        var uri = output.getOutputFiles().get("hello.txt");
        try (var is = storage.get(TenantService.MAIN_TENANT, null, uri)) {
            var content = new String(is.readAllBytes());
            assertThat(content).isEqualTo("Hello World");
        }
    }

    @EnabledIfEnvironmentVariable(named = "GOOGLE_API_KEY", matches = ".*")
    @EnabledIfEnvironmentVariable(named = "GOOGLE_CSI", matches = ".*")
    @Test
    void withGoogleCustomWebSearchContentRetriever() throws Exception {
        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "gemini-2.0-flash",
            "apiKey", GOOGLE_API_KEY,
            "csi", GOOGLE_CSI
        ));

        var agent = AIAgent.builder()
            .provider(
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .build()
            )
            .contentRetrievers(Property.ofValue(List.of(GoogleCustomWebSearch.builder()
                .csi(Property.ofExpression("{{ csi }}"))
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .build())))
            .prompt(Property.ofValue("What is the capital of France and how many people live there?"))
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = agent.run(runContext);
        assertThat(output.getTextOutput()).isNotNull();
        assertThat(output.getTextOutput()).contains("Paris");
    }

    @EnabledIfEnvironmentVariable(named = "TAVILY_API_KEY", matches = ".*")
    @Test
    void withTavilyWebSearchContentRetriever() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1",
            "apiKey", TAVILY_API_KEY
        ));
        var agent = AIAgent.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
                .contentRetrievers(Property.ofValue(List.of(TavilyWebSearch.builder()
                    .apiKey(Property.ofExpression("{{ apiKey }}"))
                    .build())))
            .prompt(Property.ofValue("What is the capital of France and how many people live there?"))
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = agent.run(runContext);
        assertThat(output.getTextOutput()).isNotNull();
        assertThat(output.getTextOutput()).contains("Paris");
    }

    /**
     * Tests end-to-end retrieval-augmented generation (RAG) using:
     * <ul>
     *     <li>Google Gemini embedding model for document ingestion</li>
     *     <li>A Kestra KV-store based embedding store</li>
     *     <li>An EmbeddingStoreRetriever for semantic search</li>
     *     <li>A Google Gemini chat model for answering the query</li>
     * </ul>
     **/
    @EnabledIfEnvironmentVariable(named = "GOOGLE_API_KEY", matches = ".*")
    @Test
    void withEmbeddingStoreRetriever() throws Exception {
        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "gemini-2.0-flash",
            "googleApiKey", GOOGLE_API_KEY
        ));

        // Ingest documents into KV Store
        var ingest = IngestDocument.builder()
            .provider(
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .modelName(Property.ofValue("gemini-embedding-exp-03-07"))
                    .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                    .build()
            )
            .embeddings(io.kestra.plugin.ai.embeddings.KestraKVStore.builder().build())
            .fromDocuments(List.of(
                IngestDocument.InlineDocument.builder()
                    .content(Property.ofValue("Paris is the capital of France with a population of over 2.1 million people"))
                    .build(),
                IngestDocument.InlineDocument.builder()
                    .content(Property.ofValue("The Eiffel Tower is the most famous landmark in Paris at 330 meters tall"))
                    .build()
            ))
            .build();

        IngestDocument.Output ingestOutput = ingest.run(runContext);
        assertThat(ingestOutput.getIngestedDocuments()).isEqualTo(2);

        // Query using EmbeddingStoreRetriever
        var agent = AIAgent.builder()
            .provider(
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                    .build()
            )
            .contentRetrievers(Property.ofValue(List.of(
                EmbeddingStoreRetriever.builder()
                    .embeddings(io.kestra.plugin.ai.embeddings.KestraKVStore.builder().build())
                    .embeddingProvider(
                        GoogleGemini.builder()
                            .type(GoogleGemini.class.getName())
                            .modelName(Property.ofValue("gemini-embedding-exp-03-07"))
                            .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                            .build()
                    )
                    .build()
            )))
            .prompt(Property.ofValue("What is the capital of France and how many people live there?"))
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = agent.run(runContext);
        assertThat(output.getTextOutput()).isNotNull();
        assertThat(output.getTextOutput()).contains("Paris");
    }

    /**
     * Integration test demonstrating how can aggregate context from
     * multiple heterogeneous retrieval sources, including:
     * <ul>
     *   <li><b>Two Kestra KVStore embedding stores</b> containing technical and business documents</li>
     *   <li><b>Google Gemini</b> for both embedding generation
     *       ({@code gemini-embedding-exp-03-07}) and LLM responses
     *       ({@code gemini-2.0-flash})</li>
     *   <li><b>Tavily Web Search</b> for real-time, general-purpose internet search</li>
     *   <li><b>Google Custom Search (CSE)</b> for domain-specific or curated web search results</li>
     * </ul>
     */

    @EnabledIfEnvironmentVariable(named = "GOOGLE_API_KEY", matches = ".*")
    @EnabledIfEnvironmentVariable(named = "GOOGLE_CSI", matches = ".*")
    @EnabledIfEnvironmentVariable(named = "TAVILY_API_KEY", matches = ".*")
    @Test
    void withMultipleEmbeddingStores_andWebSearches() throws Exception {
        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "gemini-2.0-flash",
            "googleApiKey", GOOGLE_API_KEY,
            "tavilyApiKey", TAVILY_API_KEY,
            "csi", GOOGLE_CSI
        ));

        // Ingest technical docs into KV Store 1
        var technicalIngest = IngestDocument.builder()
            .provider(
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .modelName(Property.ofValue("gemini-embedding-exp-03-07"))
                    .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                    .build()
            )
            .embeddings(io.kestra.plugin.ai.embeddings.KestraKVStore.builder()
                .kvName(Property.ofValue("technical-docs"))
                .build())
            .drop(Property.ofValue(true))
            .fromDocuments(List.of(
                IngestDocument.InlineDocument.builder()
                    .content(Property.ofValue("Kestra is an open-source orchestration platform for workflow automation"))
                    .build()
            ))
            .build();

        technicalIngest.run(runContext);

        // Ingest business docs into KV Store 2
        var businessIngest = IngestDocument.builder()
            .provider(
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .modelName(Property.ofValue("gemini-embedding-exp-03-07"))
                    .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                    .build()
            )
            .embeddings(io.kestra.plugin.ai.embeddings.KestraKVStore.builder()
                .kvName(Property.ofValue("business-docs"))
                .build())
            .drop(Property.ofValue(true))
            .fromDocuments(List.of(
                IngestDocument.InlineDocument.builder()
                    .content(Property.ofValue("We serve enterprise customers in financial services and healthcare"))
                    .build()
            ))
            .build();

        businessIngest.run(runContext);

        // Query using multiple embedding stores + multiple web searches
        var agent = AIAgent.builder()
            .provider(
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                    .build()
            )
            .contentRetrievers(Property.ofValue(List.of(
                // Embedding Store 1
                EmbeddingStoreRetriever.builder()
                    .embeddings(io.kestra.plugin.ai.embeddings.KestraKVStore.builder()
                        .kvName(Property.ofValue("technical-docs"))
                        .build())
                    .embeddingProvider(
                        GoogleGemini.builder()
                            .type(GoogleGemini.class.getName())
                            .modelName(Property.ofValue("gemini-embedding-exp-03-07"))
                            .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                            .build()
                    )
                    .build(),
                // Embedding Store 2
                EmbeddingStoreRetriever.builder()
                    .embeddings(io.kestra.plugin.ai.embeddings.KestraKVStore.builder()
                        .kvName(Property.ofValue("business-docs"))
                        .build())
                    .embeddingProvider(
                        GoogleGemini.builder()
                            .type(GoogleGemini.class.getName())
                            .modelName(Property.ofValue("gemini-embedding-exp-03-07"))
                            .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                            .build()
                    )
                    .build(),
                // Web Search 1: Tavily
                TavilyWebSearch.builder()
                    .apiKey(Property.ofExpression("{{ tavilyApiKey }}"))
                    .build(),
                // Web Search 2: Google Custom Search
                GoogleCustomWebSearch.builder()
                    .csi(Property.ofExpression("{{ csi }}"))
                    .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                    .build()
            )))
            .prompt(Property.ofValue("What is Kestra and what industries does it serve?"))
            .configuration(ChatConfiguration.builder().temperature(Property.ofValue(0.1)).seed(Property.ofValue(123456789)).build())
            .build();

        var output = agent.run(runContext);
        assertThat(output.getTextOutput()).isNotNull();
    }

    /**
     * Integration test demonstrating an {@link AIAgent} that retrieves context from multiple
     * heterogeneous sources, including:
     * <ul>
     *   <li><b>Pinecone</b> for external vector retrieval</li>
     *   <li><b>Kestra KVStore</b> as lightweight in-memory embedding stores</li>
     *   <li><b>Google Gemini</b> for both embeddings and LLM reasoning</li>
     *   <li><b>Tavily Web Search</b> for real-time external information</li>
     * </ul>
     * This test validates that the agent can combine all retrieval sources into a unified answer.
     */
    @Test
    @EnabledIfEnvironmentVariable(named = "GOOGLE_API_KEY", matches = ".*")
    @EnabledIfEnvironmentVariable(named = "TAVILY_API_KEY", matches = ".*")
    @EnabledIfEnvironmentVariable(named = "PINECONE_API_KEY", matches = ".*")
    void withMultipleDifferentEmbeddingStores_andWebSearch() throws Exception {
        RunContext runContext = runContextFactory.of("namespace", Map.of(
            "modelName", "gemini-2.0-flash",
            "googleApiKey", GOOGLE_API_KEY,
            "tavilyApiKey", TAVILY_API_KEY,
            "pineconeApiKey", PINECONE_API_KEY
        ));

        var businessIngest = IngestDocument.builder()
            .provider(
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .modelName(Property.ofValue("gemini-embedding-exp-03-07"))
                    .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                    .build()
            )
            .embeddings(io.kestra.plugin.ai.embeddings.KestraKVStore.builder()
                .kvName(Property.ofValue("qdrant-sim"))
                .build())
            .drop(Property.ofValue(true))
            .fromDocuments(List.of(
                IngestDocument.InlineDocument.builder()
                    .content(Property.ofValue("Kestra is used by Fortune 500 companies for data pipelines"))
                    .build()
            ))
            .build();

        businessIngest.run(runContext);

        var agent = AIAgent.builder()
            .provider(
                GoogleGemini.builder()
                    .type(GoogleGemini.class.getName())
                    .modelName(Property.ofExpression("{{ modelName }}"))
                    .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                    .build()
            )
            .contentRetrievers(Property.ofValue(List.of(

                /* ---------- Pinecone Retriever ---------- */
                EmbeddingStoreRetriever.builder()
                    .embeddings(io.kestra.plugin.ai.embeddings.Pinecone.builder()
                        .apiKey(Property.ofExpression("{{ pineconeApiKey }}"))
                        .index(Property.ofValue("embeddings"))
                        .cloud(Property.ofValue("aws"))
                        .region(Property.ofValue("us-east-1"))
                        .namespace(Property.ofValue("test"))  // optional
                        .build())
                    .embeddingProvider(
                        GoogleGemini.builder()
                            .type(GoogleGemini.class.getName())
                            .modelName(Property.ofValue("gemini-embedding-exp-03-07"))
                            .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                            .build()
                    )
                    .build(),

                /* ---------- KV Store #1 ---------- */
                EmbeddingStoreRetriever.builder()
                    .embeddings(io.kestra.plugin.ai.embeddings.KestraKVStore.builder()
                        .kvName(Property.ofValue("pinecone-sim"))
                        .build())
                    .embeddingProvider(
                        GoogleGemini.builder()
                            .type(GoogleGemini.class.getName())
                            .modelName(Property.ofValue("gemini-embedding-exp-03-07"))
                            .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                            .build()
                    )
                    .build(),

                /* ---------- KV Store #2 ---------- */
                EmbeddingStoreRetriever.builder()
                    .embeddings(io.kestra.plugin.ai.embeddings.KestraKVStore.builder()
                        .kvName(Property.ofValue("qdrant-sim"))
                        .build())
                    .embeddingProvider(
                        GoogleGemini.builder()
                            .type(GoogleGemini.class.getName())
                            .modelName(Property.ofValue("gemini-embedding-exp-03-07"))
                            .apiKey(Property.ofExpression("{{ googleApiKey }}"))
                            .build()
                    )
                    .build(),

                /* ---------- Tavily Web Search ---------- */
                TavilyWebSearch.builder()
                    .apiKey(Property.ofExpression("{{ tavilyApiKey }}"))
                    .build()
            )))
            .prompt(Property.ofValue("What programming languages does Kestra support and which companies use it?"))
            .configuration(ChatConfiguration.builder()
                .temperature(Property.ofValue(0.1))
                .seed(Property.ofValue(123456789))
                .build())
            .build();

        var output = agent.run(runContext);

        assertThat(output.getTextOutput()).isNotNull();
        assertThat(output.getTextOutput()).contains("Kestra");
    }

}