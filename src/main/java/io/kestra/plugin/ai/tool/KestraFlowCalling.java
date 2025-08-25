package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonNumberSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import dev.langchain4j.service.tool.ToolExecutor;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.Label;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.Execution;
import io.kestra.core.models.flows.FlowInterface;
import io.kestra.core.models.property.Property;
import io.kestra.core.queues.QueueFactoryInterface;
import io.kestra.core.queues.QueueInterface;
import io.kestra.core.runners.DefaultRunContext;
import io.kestra.core.runners.FlowMetaStoreInterface;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.core.serializers.ListOrMapOfLabelDeserializer;
import io.kestra.core.serializers.ListOrMapOfLabelSerializer;
import io.kestra.core.utils.IdUtils;
import io.kestra.core.utils.ListUtils;
import io.kestra.core.utils.MapUtils;
import io.kestra.core.validations.NoSystemLabelValidation;
import io.kestra.plugin.ai.domain.ToolProvider;
import io.micronaut.inject.qualifiers.Qualifiers;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.time.ZonedDateTime;
import java.util.*;
import java.util.stream.Collectors;

import static io.kestra.core.utils.Rethrow.throwFunction;

@Getter
@SuperBuilder
@NoArgsConstructor
@Plugin(
    examples =  {
        @Example(
            title = "Call a Kestra flow as a tool",
            full = true,
            code = {
                """
                    id: kestra-flow-tool
                    namespace: company.team

                    inputs:
                      - id: first
                        type: STRING
                        defaults: |
                          I want to execute a flow to say Hello World.

                    tasks:
                      - id: first
                        type: io.kestra.plugin.ai.completion.ChatCompletion
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-2.5-flash
                          apiKey: "{{ secret('GOOGLE_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.KestraFlowCalling
                            namespace: company.team
                            flowId: hello-world
                            # The target flow has no description so we must add one here for the LLM so it choose to call it
                            description: A flow that say Hello World
                        messages:
                          - type: SYSTEM
                            content: You are an AI agent, please use the provided tool to fulfill the request.
                          - type: USER
                            content: "{{inputs.first}}\""""
            }
        ),
        @Example(
            title = "Call a Kestra flow as a tool, passing the flow id and namespace inside the prompt",
            full = true,
            code = {
                """
                    id: kestra-flow-tool
                    namespace: company.team

                    inputs:
                      - id: first
                        type: STRING
                        defaults: |
                          I want to execute a flow id 'hello-world' from the 'company.team' namespace.

                    tasks:
                      - id: first
                        type: io.kestra.plugin.ai.completion.ChatCompletion
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-2.5-flash
                          apiKey: "{{ secret('GOOGLE_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.KestraFlowCalling
                        messages:
                          - type: SYSTEM
                            content: You are an AI agent, please use the provided tool to fulfill the request.
                          - type: USER
                            content: "{{inputs.first}}\""""
            }
        ),
        @Example(
            title = "Do sentiment analysis and summarize a product review and call a flow as a tool with the summary as input and the sentiment as label",
            full = true,
            code = {
                """
                    id: analyse-product-review
                    namespace: company.team

                    inputs:
                      - id: review
                        type: STRING
                        defaults: |
                          I tested the product and it is not good, it lack functionalities and the first time I use it, it becomes warm and made stange noise. I think I would not recommand it to anybody, except it's low price, there is nothing for it.

                    tasks:
                      - id: first
                        type: io.kestra.plugin.ai.completion.ChatCompletion
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-2.5-flash
                          apiKey: "{{ secret('GOOGLE_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.KestraFlowCalling
                            namespace: company.team
                            flowId: product-review-summary
                        messages:
                          - type: SYSTEM
                            content: You are an AI agent, call the flow that handle product reviews via the provided tool. The flow must have an input with id 'summary' which contains the summary of the review which will be passed as a user message. Also add a label to the flow named 'sentiment' with the sentiment of the review.
                          - type: USER
                            content: "{{inputs.review}}"

                ---

                    id: product-review-summary
                    namespace: company.team
                    description: Handle product review

                    inputs:
                      - id: summary
                        type: STRING

                    tasks:
                      - id: logSummary
                        type: io.kestra.plugin.core.log.Log
                        message: "This is the summary of the product review: {{inputs.summary}}"
                      - id: logSentiment
                        type: io.kestra.plugin.core.log.Log
                        message: "This is the sentiment of the review: {{labels.sentiment}}\""""
            }
        ),
    }
)
@JsonDeserialize
@Schema(
    title = "Call a Kestra flow as a tool.",
    description = """
        This tool provider will provide a tool that can call a Kestra flow.
        It can be used in two ways: by defining the flow inside the tool, or by defining it inside the LLM prompt.
        The LLM can set `inputs`, `labels`, and a `scheduledDate` if needed.
        The called flow will have the `correlationId` label of the current flow if none is provided by the LLM.

        **Flow defined inside the tool**
        In this case, the provider will create a tool named `kestra_flow_<namespace>_<flowId>` so you can add multiple time the tool for different flows and let the LLM decide which one to use..
        To instruct what the flow is doing, the provider either use the description set as the `description` property or the description of the flow itself. If none are provided an error will be thrown.

        **Flow defined inside the LLM prompt**
        In this case, the provider will create a tool named `kestra_flow`.
        The LLM must set both `namespace` and `flowId` tool parameter."""
)
public class KestraFlowCalling extends ToolProvider {
    // Tool description, it could be fine-tuned if needed
    private static final String TOOL_DEFINED_DESCRIPTION = "This tool allows to execute a Kestra workflow also called a flow. This tool will respond with the flow execution information.";
    private static final String TOOL_LLM_DESCRIPTION = """
        This tool allows to execute a Kestra workflow also called a flow. This tool will respond with the flow execution information.
        The namespace and the id of the flow must be passed as tool parameters""";

    @Schema(
        title = "Description of the flow if not already provided inside the flow itself",
        description = """
            Use it only if you define the flow inside the tool provider.
            The LLM needs a description for the tool to be able to know if it needs to use it or not.
            If the flow has a description, the tool provider will use it, otherwise this property must be used to give a description."""
    )
    private Property<String> description;

    @Schema(title = "The target flow namespace")
    private Property<String> namespace;

    @Schema(title = "The target flow id")
    private Property<String> flowId;

    @Schema(title = "The target flow revision")
    private Property<Integer> revision;

    @Schema(
        title = "The inputs to pass to the flow to be called",
        description = "Any inputs passed by the LLM will override those defined here."
    )
    @PluginProperty(dynamic = true)
    private Map<String, Object> inputs;

    @Schema(
        title = "The labels to pass to the flow to be executed",
        description = "Any labels passed by the LLM will override those defined here.",
        implementation = Object.class, oneOf = {List.class, Map.class}
    )
    @PluginProperty(dynamic = true)
    @JsonSerialize(using = ListOrMapOfLabelSerializer.class)
    @JsonDeserialize(using = ListOrMapOfLabelDeserializer.class)
    private List<@NoSystemLabelValidation Label> labels;

    @Builder.Default
    @Schema(
        title = "Whether the flow should inherit labels from this execution that triggered it",
        description = """
            By default, labels are not passed to the flow execution. If you set this option to `true`, the flow execution will inherit all labels from the this execution.
            Any labels passed by the LLM will override those defined here."""
    )
    private final Property<Boolean> inheritLabels = Property.ofValue(false);

    @Schema(
        title = "Don't trigger the flow now but schedule it on a specific date.",
        description = "If the LLM sets a scheduleDate, it will override the one defined here."
    )
    private Property<ZonedDateTime> scheduleDate;

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext) throws IllegalVariableEvaluationException {
        boolean hasDefinedFlow = this.namespace != null && this.flowId != null;
        if (this.namespace != null && this.flowId == null) {
            throw new IllegalArgumentException("Flow ID must be specified when you set the namespace");
        }
        if (this.namespace == null && this.flowId != null) {
            throw new IllegalArgumentException("Namespace must be specified when you set the flow ID");
        }

        var rInputs = runContext.render(MapUtils.emptyOnNull(inputs));

        // compute labels
        boolean rInheritedLabels = runContext.render(inheritLabels).as(Boolean.class).orElse(false);
        List<Label> executionLabels = MapUtils.nestedToFlattenMap(MapUtils.emptyOnNull((Map<String, Object>) runContext.getVariables().get("labels"))).entrySet().stream()
            .map(entry -> new Label(entry.getKey(), entry.getValue().toString()))
            .toList();
        List<Label> rLabels = ListUtils.emptyOnNull(labels).stream().map(throwFunction(label -> new Label(runContext.render(label.key()), runContext.render(label.value())))).toList();

        var jsonSchema = JsonObjectSchema.builder()
            .addProperty("labels", JsonArraySchema.builder().items(
                        JsonObjectSchema.builder()
                            .addStringProperty("key", "The label key.")
                            .addStringProperty("value", "The label value.")
                            .build()
                    ).description("The list of labels.")
                    .build()
            )
            .addProperty("scheduleDate", JsonStringSchema.builder()
                .description("""
                    The scheduled date of the flow. Use it only if the flow needs to be executed later and not immediately.
                    It should be an ISO8601 formatted zoned date time."""
                )
                .build());

        if (hasDefinedFlow) {
            var rNamespace = runContext.render(this.namespace).as(String.class).orElseThrow();
            var rFlowId = runContext.render(this.flowId).as(String.class).orElseThrow();
            var rRevision = runContext.render(this.revision).as(Integer.class);

            var defaultRunContext = (DefaultRunContext) runContext;
            var flowMetaStoreInterface = defaultRunContext.getApplicationContext().getBean(FlowMetaStoreInterface.class);
            var flowInfo = runContext.flowInfo();
            var flowInterface = flowMetaStoreInterface.findByIdFromTask(flowInfo.tenantId(), rNamespace, rFlowId, rRevision, flowInfo.tenantId(), flowInfo.namespace(), flowInfo.id())
                .orElseThrow(() -> new IllegalArgumentException("Unable to find flow at '"+ rFlowId + "' in namespace '" + rNamespace + "'"));

            var rDescription = runContext.render(this.description).as(String.class).orElse(flowInterface.getDescription());
            if (rDescription == null) {
                throw new IllegalArgumentException("You must provide a description from the tool 'description' property of the flow description");
            }

            jsonSchema.description(rDescription);
            if (!ListUtils.isEmpty(flowInterface.getInputs())) {
                jsonSchema.addProperty("inputs", JsonArraySchema.builder().items(
                            JsonObjectSchema.builder()
                                .addStringProperty("id", "The input id.")
                                .addStringProperty("value", "The input value.")
                                .build()
                        ).description("The list of inputs.")
                        .build()
                );
                // check if there are any mandatory inputs
                if (flowInterface.getInputs().stream()
                    .anyMatch(input -> input.getRequired() && input.getDefaults() == null && !rInputs.containsKey(input.getId()))) {
                    jsonSchema.required("inputs");
                }
            }

            return Map.of(
                ToolSpecification.builder()
                    .name("kestra_flow_" + IdUtils.fromPartsAndSeparator('_', flowInterface.getNamespace().replace('.', '_'), flowInterface.getId()))
                    .description(TOOL_DEFINED_DESCRIPTION)
                    .parameters(jsonSchema.build())
                    .build(),
                new KestraDefinedFlowToolExecutor(defaultRunContext, flowInterface, rInputs, rInheritedLabels, executionLabels, rLabels)
            );
        } else {
            jsonSchema.description(TOOL_LLM_DESCRIPTION);
            jsonSchema.addProperty("namespace", JsonStringSchema.builder().build());
            jsonSchema.addProperty("flowId", JsonStringSchema.builder().build());
            jsonSchema.addProperty("revision", JsonNumberSchema.builder().build());
            jsonSchema.addProperty("inputs", JsonArraySchema.builder().items(
                        JsonObjectSchema.builder()
                            .addStringProperty("id", "The input id.")
                            .addStringProperty("value", "The input value.")
                            .build()
                    ).description("The list of inputs.")
                    .build()
            );
            jsonSchema.required("namespace", "flowId");

            return Map.of(
                ToolSpecification.builder()
                    .name("kestra_flow")
                    .description(TOOL_LLM_DESCRIPTION)
                    .parameters(jsonSchema.build())
                    .build(),
                new KestraLLMFlowToolExecutor((DefaultRunContext) runContext, rInputs, rInheritedLabels, executionLabels, rLabels)
            );
        }
    }

    static class KestraDefinedFlowToolExecutor extends AbstractKestraFlowToolExecutor {
        private final FlowInterface flowInterface;

        KestraDefinedFlowToolExecutor(DefaultRunContext runContext, FlowInterface flowInterface, Map<String, Object> predefinedInputs, boolean inheritedLabels, List<Label> executionLabels, List<Label> taskLabels) {
            super(runContext, predefinedInputs, inheritedLabels, executionLabels, taskLabels);

            this.flowInterface = flowInterface;
        }


        @Override
        protected FlowInterface getFlow(Map<String, Object> parameters) {
            return flowInterface;
        }
    }

    static class KestraLLMFlowToolExecutor extends AbstractKestraFlowToolExecutor {
        private final DefaultRunContext runContext;

        KestraLLMFlowToolExecutor(DefaultRunContext runContext, Map<String, Object> predefinedInputs, boolean inheritedLabels, List<Label> executionLabels, List<Label> taskLabels) {
            super(runContext, predefinedInputs, inheritedLabels, executionLabels, taskLabels);

            this.runContext = runContext;
        }

        @Override
        protected FlowInterface getFlow(Map<String, Object> parameters) {
            var namespace = (String) parameters.get("namespace");
            var flowId = (String) parameters.get("flowId");
            var revision = Optional.ofNullable((Integer) parameters.get("revision"));

            var flowMetaStoreInterface = runContext.getApplicationContext().getBean(FlowMetaStoreInterface.class);
            var flowInfo = runContext.flowInfo();
            return flowMetaStoreInterface.findByIdFromTask(flowInfo.tenantId(), namespace, flowId, revision, flowInfo.tenantId(), flowInfo.namespace(), flowInfo.id())
                .orElseThrow(() -> new IllegalArgumentException("Unable to find flow at '"+ flowId + "' in namespace '" + namespace + "'"));
        }
    }

    static abstract class AbstractKestraFlowToolExecutor implements ToolExecutor {
        private final DefaultRunContext runContext;
        private final Map<String, Object> predefinedInputs;
        private final boolean inheritedLabels;
        private final List<Label> executionLabels;
        private final List<Label> taskLabels;

        AbstractKestraFlowToolExecutor(DefaultRunContext runContext, Map<String, Object> predefinedInputs, boolean inheritedLabels, List<Label> executionLabels, List<Label> taskLabels) {
            this.runContext = runContext;
            this.predefinedInputs = predefinedInputs;
            this.inheritedLabels = inheritedLabels;
            this.executionLabels = executionLabels;
            this.taskLabels = taskLabels;
        }

        protected abstract FlowInterface getFlow(Map<String, Object> parameters);

        @Override
        @SuppressWarnings("unchecked")
        public String execute(ToolExecutionRequest toolExecutionRequest, Object memoryId) {
            runContext.logger().debug("Tool execution request: {}", toolExecutionRequest);
            try {
                var flowParameters = JacksonMapper.toMap(toolExecutionRequest.arguments());

                var scheduledDate = Optional.ofNullable((String) flowParameters.get("scheduleDate")).map(d -> ZonedDateTime.parse(d));

                var flowInterface = getFlow(flowParameters);

                List<Label> newLabels =  inheritedLabels ? new ArrayList<>(filterLabels(executionLabels, flowInterface)) : new ArrayList<>(systemLabels(executionLabels));
                newLabels.addAll(taskLabels);

                // merge LLM provided labels with tool predefined one
                var labels = (List<Map<String, String>>) flowParameters.get("labels");
                var labelList = ListUtils.emptyOnNull(labels).stream()
                    .map(label -> new Label(label.get("key"), label.get("value")))
                    .toList();
                var predefinedLabelsToAdd = newLabels.stream().filter(l1 -> labelList.stream().noneMatch(l2 -> l1.key().equals(l2.key()))).toList();
                var finalLabels = ListUtils.concat(labelList, predefinedLabelsToAdd);

                // merge LLM provided inputs with tool predefined one
                var inputs = (List<Map<String, Object>>) flowParameters.get("inputs");
                var inputMap = ListUtils.emptyOnNull(inputs).stream().collect(Collectors.toMap(
                    input -> (String) input.get("id"),
                    input -> input.get("value")
                ));
                var finalInputs = MapUtils.merge(predefinedInputs, inputMap);
                // check mandatory inputs to fail the tool execution instead of triggering a flow that would fail anyway
                ListUtils.emptyOnNull(flowInterface.getInputs()).forEach(input -> {
                    if (input.getRequired() && input.getDefaults() == null && !finalInputs.containsKey(input.getId())) {
                        throw new IllegalArgumentException("You need to provide an input with the id '" + input.getId() + "'.");
                    }
                });

                var execution = Execution.newExecution(flowInterface, (f, e) -> finalInputs, finalLabels, scheduledDate);

                var executionQueue = (QueueInterface<Execution>) runContext.getApplicationContext().getBean(QueueInterface.class, Qualifiers.byName(QueueFactoryInterface.EXECUTION_NAMED));
                executionQueue.emit(execution);

                return JacksonMapper.ofJson().writeValueAsString(execution);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        private List<Label> filterLabels(List<Label> labels, FlowInterface flow) {
            if (ListUtils.isEmpty(flow.getLabels())) {
                return labels;
            }

            return labels.stream()
                .filter(label -> flow.getLabels().stream().noneMatch(flowLabel -> flowLabel.key().equals(label.key())))
                .toList();
        }

        private List<Label> systemLabels(List<Label> labels) {
            return labels.stream()
                .filter(label -> label.key().startsWith(Label.SYSTEM_PREFIX))
                .toList();
        }
    }
}
