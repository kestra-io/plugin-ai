package io.kestra.plugin.ai.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.loader.UrlDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.*;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.IngestionResult;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Metric;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.ListUtils;
import io.kestra.plugin.ai.domain.EmbeddingStoreProvider;
import io.kestra.plugin.ai.domain.ModelProvider;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;

import java.io.InputStream;
import java.net.URI;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.kestra.core.utils.Rethrow.throwConsumer;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Ingest documents into an embedding store",
    description = "Currently supports text documents (TXT, HTML, Markdown)."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = """
                Ingest documents into a KV embedding store.
                WARNING: the KV embedding store is for quick prototyping only; it stores embedding vectors in a KV store and loads them all into memory.
                """,
            code = """
                id: document_ingestion
                namespace: company.ai

                tasks:
                  - id: ingest
                    type: io.kestra.plugin.ai.rag.IngestDocument
                    provider:
                      type: io.kestra.plugin.ai.provider.GoogleGemini
                      modelName: gemini-embedding-exp-03-07
                      apiKey: "{{ kv('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    drop: true
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        ),
    },
    metrics = {
        @Metric(
            name = "indexed.documents",
            type = Counter.TYPE,
            unit = "records",
            description = "Number of indexed documents"
        ),
        @Metric(
            name = "input.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) input token count"
        ),
        @Metric(
            name = "output.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) output token count"
        ),
        @Metric(
            name = "total.token.count",
            type = Counter.TYPE,
            unit = "token",
            description = "Large Language Model (LLM) total token count"
        )
    },
    aliases = "io.kestra.plugin.langchain4j.rag.IngestDocument"
)
public class IngestDocument extends Task implements RunnableTask<IngestDocument.Output> {
    @Schema(
        title = "Language model provider",
        description = "Must be configured with an embedding model."
    )
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @Schema(title = "Embedding store provider")
    @NotNull
    @PluginProperty
    private EmbeddingStoreProvider embeddings;

    @Schema(
        title = "Path in the task working directory containing documents to ingest",
        description = "Each document in the directory will be ingested into the embedding store. Ingestion is recursive and protected against path traversal (CWE-22)."
    )
    private Property<String> fromPath;

    @Schema(title = "List of internal storage URIs for documents")
    @PluginProperty(internalStorageURI = true)
    private Property<List<String>> fromInternalURIs;

    @Schema(title = "List of document URLs from external sources")
    private Property<List<String>> fromExternalURLs;

    @Schema(title = "List of inline documents")
    @PluginProperty
    private List<InlineDocument> fromDocuments;

    @Schema(title = "Additional metadata to add to all ingested documents")
    private Property<Map<String, String>> metadata;

    @Schema(title = "Document splitter")
    @PluginProperty
    private DocumentSplitter documentSplitter;

    @Schema(title = "Drop the store before ingestion (useful for testing)")
    @Builder.Default
    private Property<Boolean> drop = Property.ofValue(Boolean.FALSE);

    @Schema(
        title = "Bulk ingestion size",
        description = "Maximum number of documents sent per ingestion request."
    )
    @Builder.Default
    private Property<@Min(1) Integer> bulkSize = Property.ofValue(500);

    @Override
    public Output run(RunContext runContext) throws Exception {
        List<Document> documents = new ArrayList<>();

        runContext.render(fromPath).as(String.class).ifPresent(path -> {
            // we restrict to documents on the working directory*
            // resolve protects from path traversal (CWE-22), see: https://cwe.mitre.org/data/definitions/22.html
            Path finalPath = runContext.workingDir().resolve(Path.of(path));
            documents.addAll(FileSystemDocumentLoader.loadDocumentsRecursively(finalPath));
        });

        ListUtils.emptyOnNull(fromDocuments).forEach(throwConsumer(inlineDocument -> {
            Map<String, Object> metadata = runContext.render(inlineDocument.metadata).asMap(String.class, Object.class);
            documents.add(Document.document(runContext.render(inlineDocument.content).as(String.class).orElseThrow(), Metadata.from(metadata)));
        }));

        runContext.render(fromInternalURIs).asList(String.class).forEach(throwConsumer(uri -> {
            try (InputStream file = runContext.storage().getFile(URI.create(uri))) {
                byte[] bytes = file.readAllBytes();
                documents.add(Document.from(new String(bytes)));
            }
        }));

        runContext.render(fromExternalURLs).asList(String.class).forEach(throwConsumer(url -> {
            documents.add(UrlDocumentLoader.load(url, new TextDocumentParser()));
        }));

        if (metadata != null) {
            Map<String, String> metadataMap = runContext.render(metadata).asMap(String.class, Object.class);
            documents.forEach(doc -> metadataMap.forEach((k, v) -> doc.metadata().put(k, v)));
        }

        var embeddingModel = provider.embeddingModel(runContext);
        var builder = EmbeddingStoreIngestor.builder()
            .embeddingModel(embeddingModel)
            .embeddingStore(embeddings.embeddingStore(runContext, embeddingModel.dimension(), runContext.render(drop).as(Boolean.class).orElseThrow()));

        if (documentSplitter != null) {
            builder.documentSplitter(from(documentSplitter));
        }

        EmbeddingStoreIngestor ingestor = builder.build();
        int rBulkSize = runContext.render(bulkSize).as(Integer.class).orElse(500);

        Integer inputTokenCount = null;
        Integer outputTokenCount = null;
        Integer totalTokenCount = null;

        for (int i = 0; i < documents.size(); i += rBulkSize) {
            List<Document> batch = documents.subList(i, Math.min(i + rBulkSize, documents.size()));
            IngestionResult result = ingestor.ingest(batch);

            if (result.tokenUsage() != null) {
                if (result.tokenUsage().inputTokenCount() != null) {
                    inputTokenCount = (inputTokenCount == null ? 0 : inputTokenCount) + result.tokenUsage().inputTokenCount();
                }
                if (result.tokenUsage().outputTokenCount() != null) {
                    outputTokenCount = (outputTokenCount == null ? 0 : outputTokenCount) + result.tokenUsage().outputTokenCount();
                }
                if (result.tokenUsage().totalTokenCount() != null) {
                    totalTokenCount = (totalTokenCount == null ? 0 : totalTokenCount) + result.tokenUsage().totalTokenCount();
                }
            }
        }

        runContext.metric(Counter.of("indexed.documents", documents.size()));
        if (inputTokenCount != null) {
            runContext.metric(Counter.of("input.token.count", inputTokenCount));
        }
        if (outputTokenCount != null) {
            runContext.metric(Counter.of("output.token.count", outputTokenCount));
        }
        if (totalTokenCount != null) {
            runContext.metric(Counter.of("total.token.count", totalTokenCount));
        }

        var output = Output.builder()
            .ingestedDocuments(documents.size())
            .embeddingStoreOutputs(embeddings.outputs(runContext))
            .inputTokenCount(inputTokenCount)
            .outputTokenCount(outputTokenCount)
            .totalTokenCount(totalTokenCount);

        return output.build();
    }

    private dev.langchain4j.data.document.DocumentSplitter from(DocumentSplitter splitter) {
        return switch (splitter.splitter) {
            case RECURSIVE -> DocumentSplitters.recursive(splitter.getMaxSegmentSizeInChars(), splitter.getMaxOverlapSizeInChars());
            case PARAGRAPH -> new DocumentByParagraphSplitter(splitter.getMaxSegmentSizeInChars(), splitter.getMaxOverlapSizeInChars());
            case LINE -> new DocumentByLineSplitter(splitter.getMaxSegmentSizeInChars(), splitter.getMaxOverlapSizeInChars());
            case SENTENCE -> new DocumentBySentenceSplitter(splitter.getMaxSegmentSizeInChars(), splitter.getMaxOverlapSizeInChars());
            case WORD -> new DocumentByWordSplitter(splitter.getMaxSegmentSizeInChars(), splitter.getMaxOverlapSizeInChars());
        };
    }

    @Getter
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class InlineDocument {
        @NotNull
        @Schema(title = "Document content")
        private Property<String> content;

        @Schema(title = "Document metadata")
        private Property<Map<String, Object>> metadata;
    }

    @Getter
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class DocumentSplitter {
        @NotNull
        @Builder.Default
        @Schema(
            title = "DocumentSplitter type",
            description = """
                Recommended: RECURSIVE for generic text.
                It splits into paragraphs first and fits as many as possible into a single TextSegment.
                If paragraphs are too long, they are recursively split into lines, then sentences, then words, then characters until they fit into a segment."""
        )
        private Type splitter = Type.RECURSIVE;

    @NotNull
        @Schema(title = "Maximum segment size (characters)")
        private Integer maxSegmentSizeInChars;

        @NotNull
        @Schema(title = "Maximum overlap size (characters). Only full sentences are considered for overlap.")
        private Integer maxOverlapSizeInChars;

        enum Type {
            @Schema(title = """
                Split into paragraphs first and fit as many as possible into one TextSegment.
                If paragraphs are too long, recursively split into lines, then sentences, then words, then characters until they fit.""")
            RECURSIVE,

            @Schema(title = """
                Split into paragraphs and fit as many as possible into one TextSegment.
                Paragraph boundaries are detected by at least two newline characters ("\\n\\n").""")
            PARAGRAPH,

            @Schema(title = """
                Split into lines and fit as many as possible into one TextSegment.
                Line boundaries are detected by at least one newline character ("\\n").""")
            LINE,

            @Schema(title = """
                Split into sentences and fit as many as possible into one TextSegment.
                Sentence boundaries are detected using Apache OpenNLP (English sentence model).""")
            SENTENCE,

            @Schema(title = """
                Split into words and fit as many as possible into one TextSegment.
                Word boundaries are detected by at least one space (" ").""")
            WORD
        }
    }

    @Getter
    @Builder
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Number of ingested documents")
        private Integer ingestedDocuments;

        @Schema(title = "Input token count")
        private Integer inputTokenCount;

        @Schema(title = "Output token count")
        private Integer outputTokenCount;

        @Schema(title = "Total token count")
        private Integer totalTokenCount;

        @Schema(title = "Additional outputs from the embedding store")
        private Map<String, Object> embeddingStoreOutputs;
    }

}
