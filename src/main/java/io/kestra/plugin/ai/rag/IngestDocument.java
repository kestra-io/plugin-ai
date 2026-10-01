package io.kestra.plugin.ai.rag;

import java.io.InputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

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

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.loader.UrlDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.*;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.IngestionResult;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Ingest documents into embeddings",
    description = """
        Loads text from local path, internal storage, URLs, or inline docs; splits with the chosen splitter; then writes embeddings to the configured store. `drop=true` clears the store first. Default splitter is absent; provide one for chunking."""
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
                      modelName: gemini-embedding-001
                      apiKey: "{{ secret('GEMINI_API_KEY') }}"
                    embeddings:
                      type: io.kestra.plugin.ai.embeddings.KestraKVStore
                    drop: true
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/content/blogs/release-0-24.md
                """
        ),
        @Example(
            full = true,
            title = "Attach the same metadata to every ingested document, so it can later be used to filter search results.",
            code = """
                id: document_ingestion_with_metadata
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
                    metadata:
                      source: manual-run
                      team: data
                    fromExternalURLs:
                      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/README.md
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
        ),
        @Metric(
            name = "ai.provider.calls",
            type = Counter.TYPE,
            unit = "calls",
            description = "Number of times an embedding model is obtained from a provider, tagged by provider class name"
        ),
        @Metric(
            name = "ai.embedding.store.calls",
            type = Counter.TYPE,
            unit = "calls",
            description = "Number of times an embedding store is used, tagged by store class name"
        )
    },
    aliases = "io.kestra.plugin.langchain4j.rag.IngestDocument"
)
public class IngestDocument extends Task implements RunnableTask<IngestDocument.Output> {
    private static final Set<Class<?>> SUPPORTED_METADATA_TYPES =
        Set.of(String.class, UUID.class, Integer.class, Long.class, Float.class, Double.class);

    @Schema(
        title = "Language model provider",
        description = "Model provider used to embed the documents, which must be configured with an embedding model. Use the same model at query time, or the vectors will not match. No default: this property is required.",
        example = "{type: \"io.kestra.plugin.ai.provider.GoogleGemini\", apiKey: \"{{ secret('GEMINI_API_KEY') }}\", modelName: \"gemini-embedding-001\"}"
    )
    @NotNull
    @PluginProperty(group = "main")
    private ModelProvider provider;

    @Schema(
        title = "Embedding store provider",
        description = "Vector store the embedded documents are written to. No default: this property is required.",
        example = "{type: \"io.kestra.plugin.ai.embeddings.KestraKVStore\"}"
    )
    @NotNull
    @PluginProperty(group = "main")
    private EmbeddingStoreProvider embeddings;

    @Schema(
        title = "Source directory path",
        description = "Directory in the task working directory whose documents are ingested. Traversal is recursive and guarded against path traversal (CWE-22). Not set by default; combine it freely with the other `from*` sources.",
        example = "documents"
    )
    @PluginProperty(group = "source")
    private Property<String> fromPath;

    @Schema(
        title = "Source internal storage URIs",
        description = "Kestra internal storage URIs of the documents to ingest, typically produced by an upstream task. Not set by default; combine it freely with the other `from*` sources.",
        example = "[\"{{ outputs.download.uri }}\"]"
    )
    @PluginProperty(internalStorageURI = true, group = "connection")
    private Property<List<String>> fromInternalURIs;

    @Schema(
        title = "Source external URLs",
        description = "Public URLs the documents are downloaded from before ingestion. Not set by default; combine it freely with the other `from*` sources.",
        example = "[\"https://kestra.io/docs/index.html\"]"
    )
    @PluginProperty(group = "connection")
    private Property<List<String>> fromExternalURLs;

    @Schema(
        title = "Inline documents",
        description = "Documents supplied directly in the flow, each with its own `content` and optional `metadata`. Not set by default; combine it freely with the other `from*` sources.",
        example = "[{content: \"Refunds are accepted within 30 days.\", metadata: {source: \"policy\"}}]"
    )
    @PluginProperty(group = "source")
    private List<InlineDocument> fromDocuments;

    @Schema(
        title = "Additional metadata",
        description = "Metadata attached to every ingested document, whatever its source. Existing metadata always wins on a key collision: the `metadata` of an inline document, and the keys injected by the document loader such as `file_name` and `absolute_directory_path`, are never overwritten. Supported value types are String, UUID, Integer, Long, Float and Double; `null` values are ignored, and any other type (a boolean, a list, a map, or a number outside those ranges such as a YAML big integer) is rejected before ingestion starts, so quote such values to send them as strings. From a flow a value can only be a String, Integer, Long or Double; UUID and Float exist for parity with the underlying model and are reachable only programmatically. Not set by default.",
        example = "{source: \"handbook\", version: \"2026.1\"}"
    )
    @PluginProperty(group = "advanced")
    private Property<Map<String, Object>> metadata;

    @Schema(
        title = "Document splitter",
        description = "How each document is chunked into segments before embedding. Not set by default, in which case documents are ingested without additional splitting.",
        example = "{splitter: \"RECURSIVE\", maxSegmentSizeInChars: 1000, maxOverlapSizeInChars: 200}"
    )
    @PluginProperty(group = "advanced")
    private DocumentSplitter documentSplitter;

    @Schema(
        title = "Drop the store before ingestion",
        description = "If `true`, wipe the embedding store before ingesting, which is useful when re-indexing or testing. Defaults to `false`.",
        example = "true"
    )
    @Builder.Default
    @PluginProperty(group = "advanced")
    private Property<Boolean> drop = Property.ofValue(Boolean.FALSE);

    @Schema(
        title = "Bulk ingestion size",
        description = "Maximum number of documents sent per ingestion request; lower it if the embedding provider rejects large batches. Must be at least `1`. Defaults to `500`.",
        example = "500"
    )
    @Builder.Default
    private Property<@Min(1) Integer> bulkSize = Property.ofValue(500);

    @Override
    public Output run(RunContext runContext) throws Exception {
        String rFromPath = runContext.render(fromPath).as(String.class).orElse(null);
        int rBulkSize = runContext.render(bulkSize).as(Integer.class).orElse(500);
        Map<String, Object> rMetadata = validateMetadata(runContext.render(metadata).asMap(String.class, Object.class));

        var embeddingModel = provider.embeddingModel(runContext);
        runContext.metric(Counter.of("ai.provider.calls", 1, "provider", provider.getClass().getName()));
        var embeddingStore = embeddings.embeddingStore(
            runContext,
            embeddingModel.dimension(),
            runContext.render(drop).as(Boolean.class).orElseThrow()
        );
        runContext.metric(Counter.of("ai.embedding.store.calls", 1, "store", embeddings.getClass().getName()));

        var builder = EmbeddingStoreIngestor.builder()
            .embeddingModel(embeddingModel)
            .embeddingStore(embeddingStore);

        if (documentSplitter != null) {
            builder.documentSplitter(from(documentSplitter));
        }

        EmbeddingStoreIngestor ingestor = builder.build();

        List<Document> batch = new ArrayList<>(rBulkSize);

        Counters counters = new Counters();

        if (rFromPath != null) {
            Path finalPath = runContext.workingDir().resolve(Path.of(rFromPath));
            List<Document> docs = FileSystemDocumentLoader.loadDocumentsRecursively(finalPath);

            for (Document doc : docs) {
                applyMetadata(doc, rMetadata);
                batch.add(doc);
                if (batch.size() >= rBulkSize) {
                    flushBatch(batch, ingestor, counters);
                }
            }
        }

        for (InlineDocument inlineDocument : ListUtils.emptyOnNull(fromDocuments)) {
            Map<String, Object> metadataMap = runContext.render(inlineDocument.metadata).asMap(String.class, Object.class);

            Document doc = Document.document(
                runContext.render(inlineDocument.content).as(String.class).orElseThrow(),
                Metadata.from(metadataMap)
            );

            applyMetadata(doc, rMetadata);
            batch.add(doc);
            if (batch.size() >= rBulkSize) {
                flushBatch(batch, ingestor, counters);
            }
        }

        for (String uri : runContext.render(fromInternalURIs).asList(String.class)) {
            try (InputStream file = runContext.storage().getFile(URI.create(uri))) {
                byte[] bytes = file.readAllBytes();
                var doc = Document.from(new String(bytes, StandardCharsets.UTF_8));
                applyMetadata(doc, rMetadata);
                batch.add(doc);
            }

            if (batch.size() >= rBulkSize) {
                flushBatch(batch, ingestor, counters);
            }
        }

        for (String url : runContext.render(fromExternalURLs).asList(String.class)) {
            var doc = UrlDocumentLoader.load(url, new TextDocumentParser());
            applyMetadata(doc, rMetadata);
            batch.add(doc);

            if (batch.size() >= rBulkSize) {
                flushBatch(batch, ingestor, counters);
            }
        }

        if (!batch.isEmpty()) {
            flushBatch(batch, ingestor, counters);
        }

        runContext.metric(Counter.of("indexed.documents", counters.documents));

        if (counters.input > 0) {
            runContext.metric(Counter.of("input.token.count", counters.input));
        }
        if (counters.output > 0) {
            runContext.metric(Counter.of("output.token.count", counters.output));
        }
        if (counters.total > 0) {
            runContext.metric(Counter.of("total.token.count", counters.total));
        }

        return Output.builder()
            .ingestedDocuments(counters.documents)
            .embeddingStoreOutputs(embeddings.outputs(runContext))
            .inputTokenCount(counters.input == 0 ? null : counters.input)
            .outputTokenCount(counters.output == 0 ? null : counters.output)
            .totalTokenCount(counters.total == 0 ? null : counters.total)
            .build();
    }

    /** Adds the top-level metadata to a document, keeping any value already set by the loader or by the inline document. */
    private static void applyMetadata(Document document, Map<String, Object> base) {
        if (base.isEmpty()) {
            return;
        }

        var missing = new LinkedHashMap<String, Object>();
        base.forEach((key, value) -> {
            if (!document.metadata().containsKey(key)) {
                missing.put(key, value);
            }
        });

        if (!missing.isEmpty()) {
            document.metadata().putAll(missing);
        }
    }

    /** Fails the whole task before any ingestion happens, so an unsupported value cannot leave a partially ingested store. */
    private static Map<String, Object> validateMetadata(Map<String, Object> rendered) {
        if (rendered.isEmpty()) {
            return Map.of();
        }

        var validated = new LinkedHashMap<String, Object>();
        for (var entry : rendered.entrySet()) {
            var value = entry.getValue();
            if (value == null) {
                continue;
            }
            if (!SUPPORTED_METADATA_TYPES.contains(value.getClass())) {
                throw new IllegalArgumentException(
                    "The metadata key '" + entry.getKey() + "' has an unsupported value type '" + value.getClass().getSimpleName()
                        + "' — use one of String, UUID, Integer, Long, Float or Double, or quote the value to send it as a String."
                );
            }
            validated.put(entry.getKey(), value);
        }

        return validated;
    }

    private void flushBatch(List<Document> batch, EmbeddingStoreIngestor ingestor, Counters counters) {
        if (batch.isEmpty()) {
            return;
        }

        int size = batch.size();
        IngestionResult result = ingestor.ingest(batch);
        batch.clear();

        counters.documents += size;

        var usage = result.tokenUsage();
        if (usage != null) {
            if (usage.inputTokenCount() != null) {
                counters.input += usage.inputTokenCount();
            }
            if (usage.outputTokenCount() != null) {
                counters.output += usage.outputTokenCount();
            }
            if (usage.totalTokenCount() != null) {
                counters.total += usage.totalTokenCount();
            }
        }
    }

    private static class Counters {
        int documents;
        int input;
        int output;
        int total;
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
        @Schema(
            title = "Document content",
            description = "Text of the inline document to ingest. No default: this property is required on each inline document.",
            example = "Refunds are accepted within 30 days of purchase."
        )
        private Property<String> content;

        @Schema(
            title = "Document metadata",
            description = "Metadata attached to this inline document, which takes precedence over the task-level `metadata` on key collisions. Not set by default.",
            example = "{source: \"policy\", section: \"refunds\"}"
        )
        @PluginProperty(group = "advanced")
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
            title = "Splitter type",
            description = "Granularity at which documents are chunked: `RECURSIVE`, `PARAGRAPH`, `LINE`, `SENTENCE` or `WORD`. `RECURSIVE` is recommended for generic text, as it packs whole paragraphs into a segment and only falls back to lines, sentences, words and characters when a paragraph does not fit. Defaults to `RECURSIVE`.",
            example = "RECURSIVE"
        )
        private Type splitter = Type.RECURSIVE;

        @NotNull
        @Schema(
            title = "Maximum segment size (characters)",
            description = "Largest size, in characters, of a single text segment sent to the embedding model. No default: this property is required when a `documentSplitter` is configured.",
            example = "1000"
        )
        private Integer maxSegmentSizeInChars;

        @NotNull
        @Schema(
            title = "Maximum overlap size (characters)",
            description = "Number of characters repeated between consecutive segments, which preserves context across chunk boundaries. Only whole sentences are used for the overlap. No default: this property is required when a `documentSplitter` is configured.",
            example = "200"
        )
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
        @Schema(
            title = "Ingested documents",
            description = "Number of documents written to the embedding store by this task run.",
            example = "42"
        )
        private Integer ingestedDocuments;

        @Schema(
            title = "Input token count",
            description = "Tokens consumed embedding the documents, when the provider reports them.",
            example = "15320"
        )
        private Integer inputTokenCount;

        @Schema(
            title = "Output token count",
            description = "Tokens produced by the embedding model, when the provider reports them. Usually `0`, since embedding models return vectors rather than text.",
            example = "0"
        )
        private Integer outputTokenCount;

        @Schema(
            title = "Total token count",
            description = "Sum of input and output tokens billed for the ingestion, when the provider reports them.",
            example = "15320"
        )
        private Integer totalTokenCount;

        @Schema(
            title = "Embedding store outputs",
            description = "Extra outputs the embedding store reports after ingestion, whose keys depend on the store implementation.",
            example = "{kvName: \"my_flow-embedding-store\"}"
        )
        private Map<String, Object> embeddingStoreOutputs;
    }

}
