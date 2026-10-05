package io.kestra.plugin.ai.domain;

import java.net.URI;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.time.StopWatch;

import com.fasterxml.jackson.core.JsonProcessingException;

import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.core.utils.ListUtils;
import io.kestra.plugin.ai.AIUtils;
import io.kestra.plugin.ai.provider.TimingChatModelListener;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.service.Result;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import lombok.experimental.SuperBuilder;

import static io.kestra.core.utils.Rethrow.throwFunction;

@SuperBuilder
@Getter
public class AIOutput implements io.kestra.core.models.tasks.Output {
    @Schema(
        title = "Text output",
        description = "Text the model generated. Populated when the response format is `TEXT` (the default), and null otherwise.",
        example = "The capital of France is Paris."
    )
    @PluginProperty(group = "destination")
    private String textOutput;

    @Schema(
        title = "JSON output",
        description = "Structured object the model generated. Populated when the response format is `JSON`, and null otherwise.",
        example = "{\"category\": \"BILLING\", \"priority\": \"HIGH\"}"
    )
    @PluginProperty(group = "destination")
    private Map<String, Object> jsonOutput;

    @Schema(
        title = "Token usage",
        description = "Input, output and total tokens billed for the call, summed across every model call the task made, when the provider reports them.",
        example = "{inputTokenCount: 320, outputTokenCount: 48, totalTokenCount: 368}"
    )
    @PluginProperty(group = "advanced")
    private TokenUsage tokenUsage;

    @Schema(
        title = "Finish reason",
        description = "Why the model stopped generating, such as `STOP`, `LENGTH` or `CONTENT_FILTER`, when the provider reports it.",
        example = "STOP"
    )
    @PluginProperty(group = "advanced")
    private FinishReason finishReason;

    @Schema(
        title = "Tool executions",
        description = "Tools the model called while producing the answer, each with its arguments and result, in call order. Empty when no tool was used.",
        example = "[{requestName: \"searchWeb\", requestArguments: {query: \"Kestra docs\"}, result: \"...\"}]"
    )
    @PluginProperty(group = "advanced")
    private List<ToolExecution> toolExecutions;

    @Schema(
        title = "Intermediate responses",
        description = "Model responses produced at each step of the tool loop before the final answer, useful for debugging agent behavior.",
        example = "[{completion: \"I should search the web first.\", finishReason: \"TOOL_EXECUTION\"}]"
    )
    @PluginProperty(group = "advanced")
    private List<AIResponse> intermediateResponses;

    @Schema(
        title = "Request duration",
        description = "Wall-clock time in milliseconds spent on the model calls made by this task.",
        example = "1842"
    )
    @PluginProperty(group = "execution")
    private Long requestDuration;

    @Schema(
        title = "Output file URIs",
        description = "Kestra internal storage URIs of the files the task produced, keyed by file name.",
        example = "{report.md: \"kestra:///company/team/agent/executions/abc123/tasks/agent/report.md\"}"
    )
    @PluginProperty(additionalProperties = URI.class, group = "destination")
    private final Map<String, URI> outputFiles;

    @Schema(
        title = "Thinking output",
        description = "The model's internal reasoning text, including chain-of-thought style intermediate steps. Populated only when the model supports thinking and `configuration.returnThinking` is enabled; null otherwise.",
        example = "The user is asking about refunds, so I should check the policy document first."
    )
    @PluginProperty(group = "advanced")
    private final String thinking;

    @Schema(
        title = "Content sources",
        description = "Text segments the content retrievers injected into the context, with their metadata, so an answer can be traced back to the documents it came from. Empty when no retriever ran.",
        example = "[{content: \"Refunds are accepted within 30 days.\", metadata: {source: \"policy.pdf\"}}]"
    )
    @PluginProperty(group = "advanced")
    private final List<ContentSource> sources;

    @Schema(
        title = "Guardrail violated",
        description = "Whether an input or output guardrail expression evaluated to `false`. When `true`, `guardrailViolationMessage` holds the rule's message and no LLM output is available. Defaults to `false`.",
        example = "false"
    )
    @Builder.Default
    @PluginProperty(group = "advanced")
    private final boolean guardrailViolated = false;

    @Schema(
        title = "Guardrail violation message",
        description = "Message from the first guardrail rule that failed. Null when no guardrail was violated.",
        example = "Response leaked confidential content."
    )
    @PluginProperty(group = "advanced")
    private final String guardrailViolationMessage;

    // WARNING: When adding additional properties here, don't forget to update completion and rag ChatCompletion.Output

    public static AIOutputBuilder<?, ?> builderFrom(RunContext runContext, Result<AiMessage> result, ResponseFormatType responseFormatType) throws JsonProcessingException {
        return AIOutput.builder()
            .textOutput(responseFormatType == ResponseFormatType.TEXT ? result.content().text() : null)
            .jsonOutput(responseFormatType == ResponseFormatType.JSON ? JacksonMapper.toMap(result.content().text()) : null)
            .tokenUsage(TokenUsage.from(result.tokenUsage()))
            .finishReason(result.finishReason())
            .toolExecutions(
                ListUtils.emptyOnNull(result.toolExecutions()).stream()
                    .map(throwFunction(toolExecution -> ToolExecution.from(toolExecution)))
                    .toList()
            )
            .intermediateResponses(
                ListUtils.emptyOnNull(result.intermediateResponses()).stream()
                    .map(throwFunction(resp -> AIResponse.from(runContext, resp)))
                    .toList()
            )
            .thinking(result.content().thinking())
            .sources(
                ListUtils.emptyOnNull(result.sources()).stream()
                    .map(throwFunction(ContentSource::from))
                    .toList()
            )
            .requestDuration(extractTiming(runContext, result.finalResponse().id()));
    }

    public static AIOutput from(RunContext runContext, Result<AiMessage> result, ResponseFormatType responseFormatType) throws JsonProcessingException {
        return builderFrom(runContext, result, responseFormatType)
            .build();
    }

    private static Long extractTiming(RunContext runContext, String id) {
        if (id == null) {
            runContext.logger().info("The model provider doesn't include any identifier in its responses, thus timing the response is currently not possible");
            return null;
        }
        StopWatch timer = TimingChatModelListener.getTimer(id);
        if (timer == null) {
            runContext.logger().warn("No timer found for response id '{}'; request duration unavailable", id);
            return null;
        }
        return timer.getTime(TimeUnit.MILLISECONDS);
    }

    @Builder
    @Getter
    public static class ToolExecution {
        @Schema(
            title = "Request ID",
            description = "Identifier the model assigned to this tool execution request.",
            example = "call_a1b2c3d4"
        )
        private String requestId;

        @Schema(
            title = "Request name",
            description = "Name of the tool the model invoked.",
            example = "searchWeb"
        )
        private String requestName;

        @Schema(
            title = "Request arguments",
            description = "Arguments the model passed to the tool, parsed from its request.",
            example = "{query: \"Kestra documentation\"}"
        )
        private Map<String, Object> requestArguments;

        @Schema(
            title = "Result",
            description = "Value the tool returned, which is fed back to the model.",
            example = "Kestra is an open-source orchestration platform."
        )
        private String result;

        public static ToolExecution from(dev.langchain4j.service.tool.ToolExecution toolExecution) throws JsonProcessingException {
            return ToolExecution.builder()
                .requestId(toolExecution.request().id())
                .requestName(toolExecution.request().name())
                .requestArguments(AIUtils.parseJson(toolExecution.request().arguments()))
                .result(toolExecution.result())
                .build();
        }
    }

    @Builder
    @Getter
    public static class ContentSource {
        @Schema(
            title = "Extracted text segment",
            description = "Snippet of retrieved text relevant to the query, typically a sentence or paragraph.",
            example = "Refunds are accepted within 30 days of purchase."
        )
        @PluginProperty(group = "advanced")
        private String content;

        @Schema(
            title = "Source metadata",
            description = "Context about where the retrieved content came from, such as a URL, document title or file name.",
            example = "{source: \"policy.pdf\", page: 4}"
        )
        @PluginProperty(group = "advanced")
        private Map<String, Object> metadata;

        public static ContentSource from(Content content) {
            final TextSegment textSegment = content.textSegment();

            final ContentSourceBuilder builder = ContentSource.builder()
                .content(textSegment.text());
            if (Objects.nonNull(textSegment.metadata())) {
                builder.metadata(textSegment.metadata().toMap());
            }

            return builder.build();
        }
    }

    @Getter
    @Builder
    public static class AIResponse {
        @Schema(
            title = "Response identifier",
            description = "Identifier the provider assigned to this intermediate response.",
            example = "chatcmpl-a1b2c3d4"
        )
        @PluginProperty(group = "advanced")
        private String id;

        @Schema(
            title = "Generated text completion",
            description = "Text the model generated at this step of the tool loop.",
            example = "I should search the web before answering."
        )
        @PluginProperty(group = "advanced")
        private String completion;

        @Schema(
            title = "Token usage",
            description = "Input, output and total tokens billed for this individual model call, when the provider reports them.",
            example = "{inputTokenCount: 120, outputTokenCount: 18, totalTokenCount: 138}"
        )
        @PluginProperty(group = "advanced")
        private TokenUsage tokenUsage;

        @Schema(
            title = "Finish reason",
            description = "Why the model stopped generating at this step, such as `STOP`, `LENGTH` or `TOOL_EXECUTION`, when the provider reports it.",
            example = "TOOL_EXECUTION"
        )
        @PluginProperty(group = "advanced")
        private FinishReason finishReason;

        @Schema(
            title = "Tool execution requests",
            description = "Tool calls the model asked for at this step, before they were executed.",
            example = "[{id: \"call_a1b2c3d4\", name: \"searchWeb\", arguments: {query: \"Kestra docs\"}}]"
        )
        @PluginProperty(group = "advanced")
        private List<ToolExecutionRequest> toolExecutionRequests;

        @Schema(
            title = "Request duration",
            description = "Wall-clock time in milliseconds spent on this individual model call.",
            example = "612"
        )
        @PluginProperty(group = "execution")
        private Long requestDuration;

        static AIResponse from(RunContext runContext, ChatResponse chatResponse) throws JsonProcessingException {
            return AIResponse.builder()
                .id(chatResponse.id())
                .completion(chatResponse.aiMessage().text())
                .tokenUsage(TokenUsage.from(chatResponse.tokenUsage()))
                .finishReason(chatResponse.finishReason())
                .toolExecutionRequests(
                    ListUtils.emptyOnNull(chatResponse.aiMessage().toolExecutionRequests()).stream()
                        .map(throwFunction(req -> ToolExecutionRequest.from(req)))
                        .toList()
                )
                .requestDuration(extractTiming(runContext, chatResponse.id()))
                .build();
        }

        @Getter
        @Builder
        public static class ToolExecutionRequest {
            @Schema(
                title = "Tool execution request identifier",
                description = "Identifier the model assigned to this tool call, used to match it with its result.",
                example = "call_a1b2c3d4"
            )
            @PluginProperty(group = "advanced")
            private String id;

            @Schema(
                title = "Tool name",
                description = "Name of the tool the model asked to call.",
                example = "searchWeb"
            )
            @PluginProperty(group = "advanced")
            private String name;

            @Schema(
                title = "Tool request arguments",
                description = "Arguments the model supplied for the tool call.",
                example = "{query: \"Kestra documentation\"}"
            )
            @PluginProperty(group = "advanced")
            private Map<String, Object> arguments;

            static ToolExecutionRequest from(dev.langchain4j.agent.tool.ToolExecutionRequest toolExecutionRequest) throws JsonProcessingException {
                return ToolExecutionRequest.builder()
                    .id(toolExecutionRequest.id())
                    .name(toolExecutionRequest.name())
                    .arguments(AIUtils.parseJson(toolExecutionRequest.arguments()))
                    .build();

            }
        }
    }

    // This is a hack to make JavaDoc working as annotation processor didn't run before JavaDoc.
    // See https://stackoverflow.com/questions/51947791/javadoc-cannot-find-symbol-error-when-using-lomboks-builder-annotation
    public static abstract class AIOutputBuilder<C extends AIOutput, B extends AIOutput.AIOutputBuilder<C, B>> {
    }
}
