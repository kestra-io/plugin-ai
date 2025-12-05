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
import io.kestra.plugin.ai.provider.Ollama;
import io.kestra.plugin.ai.provider.OpenAI;
import io.kestra.plugin.ai.rag.ChatCompletion;
import io.kestra.plugin.ai.rag.IngestDocument;
import io.kestra.plugin.ai.retriever.GoogleCustomWebSearch;
import io.kestra.plugin.ai.retriever.TavilyWebSearch;
import io.kestra.plugin.ai.tool.DockerMcpClient;
import io.kestra.plugin.ai.tool.StdioMcpClient;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.util.List;
import java.util.Map;

import static io.kestra.plugin.ai.ContainerTest.ollamaEndpoint;
import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class AIAgentTest {
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
}