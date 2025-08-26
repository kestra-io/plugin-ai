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
            title = "Call a Kestra flow as a tool, explicitly defining the flow id and namespace in the tool definition",
            full = true,
            code = {
                """
                    id: agent_calling_flows_explicitly
                    namespace: company.ai

                    inputs:
                      - id: use_case
                        type: SELECT
                        description: Your Orchestration Use Case
                        defaults: Hello World
                        values:
                          - Business Automation
                          - Business Processes
                          - Data Engineering Pipeline
                          - Data Warehouse and Analytics
                          - Infrastructure Automation
                          - Microservices and APIs
                          - Hello World

                    tasks:
                      - id: agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        prompt: Execute a flow that best matches the {{ inputs.use_case }} use case selected by the user
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-2.5-flash
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: business-automation
                            description: Business Automation

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: business-processes
                            description: Business Processes

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: data-engineering-pipeline
                            description: Data Engineering Pipeline

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: dwh-and-analytics
                            description: Data Warehouse and Analytics

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: file-processing
                            description: File Processing

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: hello-world
                            description: Hello World

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: infrastructure-automation
                            description: Infrastructure Automation

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: microservices-and-apis
                            description: Microservices and APIs"""
            }
        ),
        @Example(
            title = "Call a Kestra flow as a tool, implicitly passing the flow id and namespace in the prompt",
            full = true,
            code = {
                """
                    id: agent_calling_flows_implicitly
                    namespace: company.ai

                    inputs:
                      - id: use_case
                        type: SELECT
                        description: Your Orchestration Use Case
                        defaults: Hello World
                        values:
                          - Business Automation
                          - Business Processes
                          - Data Engineering Pipeline
                          - Data Warehouse and Analytics
                          - Infrastructure Automation
                          - Microservices and APIs
                          - Hello World

                    tasks:
                      - id: agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        prompt: |
                          Execute a flow that best matches the {{ inputs.use_case }} use case selected by the user. Use the following mapping of use cases to flow ids:
                          - Business Automation: business-automation
                          - Business Processes: business-processes
                          - Data Engineering Pipeline: data-engineering-pipeline
                          - Data Warehouse and Analytics: dwh-and-analytics
                          - Infrastructure Automation: infrastructure-automation
                          - Microservices and APIs: microservices-and-apis
                          - Hello World: hello-world
                          Remember that all those flows are in the tutorial namespace.
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-2.5-flash
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.KestraFlow"""
            }
        ),
    }
)
@JsonDeserialize
@Schema(
    title = "Call a Kestra flow as a tool.",
    description = """
        This tool allows an LLM to call a Kestra flow.

        It supports two usage modes:

        **1. Call a flow explicitly defined in the tool specification**
        In this mode, the AI Agent creates a tool named `kestra_flow_<namespace>_<flowId>`.
        Multiple flows can be added as separate tools, and the LLM can choose which one to call.
        The tool's description comes from the tool's `description` property or the flow's description.
        If no description is available, an error will be raised.

        **2. Call a flow defined in the LLM prompt**
        In this mode, the AI Agent creates a single tool named `kestra_flow`.
        The LLM will infer the `namespace` and `flowId` parameters from the prompt.

        The LLM can also set `inputs`, `labels`, and `scheduledDate` if required.
        If no `correlationId` is provided, the called flow will inherit `correlationId` from the agent's execution."""
)
public class KestraFlow extends ToolProvider {
    // Tool description, it could be fine-tuned if needed
    private static final String TOOL_DEFINED_DESCRIPTION = "This tool allows to execute a Kestra flow and output the execution details.";
    private static final String TOOL_LLM_DESCRIPTION = """
        This tool allows to execute a Kestra workflow also called a flow. This tool will respond with the flow execution information.
        The namespace and the id of the flow must be passed as tool parameters""";

    @Schema(
        title = "Description of the flow if not already provided inside the flow itself",
        description = """
            Use it only if you define the flow in the tool definition.
            The LLM needs a tool description to identify whether to call it.
            If the flow has a description, the tool will use it. Otherwise, the description property must be explicitly defined."""
    )
    private Property<String> description;

    @Schema(title = "Namespace of the flow that should be called")
    private Property<String> namespace;

    @Schema(title = "Flow ID of the flow that should be called")
    private Property<String> flowId;

    @Schema(title = "Revision of the flow that should be called")
    private Property<Integer> revision;

    @Schema(
        title = "Input values that should be passed to flow's execution",
        description = "Any inputs passed by the LLM will override those defined here."
    )
    @PluginProperty(dynamic = true)
    private Map<String, Object> inputs;

    @Schema(
        title = "Labels that should be added to the flow's execution",
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
            By default, labels are not inherited. If you set this option to `true`, the flow execution will inherit all labels from the agent's execution.
            Any labels passed by the LLM will override those defined here."""
    )
    private final Property<Boolean> inheritLabels = Property.ofValue(false);

    @Schema(
        title = "Schedule the flow execution at a later date",
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
                throw new IllegalArgumentException("You must provide a description in the tool's description property or in the flow description");
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
                .orElseThrow(() -> new IllegalArgumentException("Unable to find flow '"+ flowId + "' in namespace '" + namespace + "'"));
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
