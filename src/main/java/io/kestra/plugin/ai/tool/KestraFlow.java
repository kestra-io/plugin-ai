package io.kestra.plugin.ai.tool;

import java.time.OffsetDateTime;
import java.time.ZonedDateTime;
import java.util.*;
import java.util.stream.Collectors;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.Label;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.SDK;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.core.serializers.ListOrMapOfLabelDeserializer;
import io.kestra.core.serializers.ListOrMapOfLabelSerializer;
import io.kestra.core.utils.IdUtils;
import io.kestra.core.utils.ListUtils;
import io.kestra.core.utils.MapUtils;
import io.kestra.core.validations.NoSystemLabelValidation;
import io.kestra.plugin.ai.domain.ToolProvider;
import io.kestra.sdk.KestraClient;
import io.kestra.sdk.internal.ApiException;
import io.kestra.sdk.model.FlowWithSource;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonNumberSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import dev.langchain4j.service.tool.ToolExecutor;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import static io.kestra.core.utils.Rethrow.throwFunction;

@Getter
@SuperBuilder
@NoArgsConstructor
@Plugin(
    examples = {
        @Example(
            title = "Call a Kestra flow as a tool, explicitly defining the flow ID and namespace in the tool definition",
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
            title = "Call a Kestra flow as a tool, implicitly passing the flow ID and namespace in the prompt",
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
                          Execute a flow that best matches the {{ inputs.use_case }} use case selected by the user. Use the following mapping of use cases to flow IDs:
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
    title = "Execute Kestra flows from an agent",
    description = """
        Triggers Kestra flows as tools, either predefined (`kestra_flow_<namespace>_<flowId>`) or generic (`kestra_flow` with namespace/flowId provided by the prompt). A description is mandatory from the flow or the tool `description`; inputs, labels, and schedule provided by the LLM override tool defaults. Labels are not inherited unless `inheritLabels=true`, while the correlationId is inherited when none is supplied."""
)
public class KestraFlow extends ToolProvider {
    // Tool description, it could be fine-tuned if needed
    private static final String TOOL_DEFINED_DESCRIPTION = "This tool executes a Kestra flow and outputs the execution details.";
    private static final String TOOL_LLM_DESCRIPTION = """
        This tool executes a Kestra workflow, also called a flow. This tool will respond with the flow execution information.
        The namespace and the ID of the flow must be passed as tool parameters.""";

    @Schema(
        title = "Description of the flow if not already provided inside the flow itself",
        description = """
            Use it only if you define the flow in the tool definition.
            The LLM needs a tool description to identify whether to call it.
            If the flow has a description, the tool will use it. Otherwise, the description property must be explicitly defined."""
    )
    @PluginProperty(group = "advanced")
    private Property<String> description;

    @Schema(title = "Namespace of the flow that should be called")
    @PluginProperty(group = "connection")
    private Property<String> namespace;

    @Schema(title = "Flow ID of the flow that should be called")
    @PluginProperty(group = "advanced")
    private Property<String> flowId;

    @Schema(title = "Revision of the flow that should be called")
    @PluginProperty(group = "advanced")
    private Property<Integer> revision;

    @Schema(
        title = "Input values that should be passed to flow's execution",
        description = "Any inputs passed by the LLM will override those defined here."
    )
    @PluginProperty(dynamic = true, group = "advanced")
    private Map<String, Object> inputs;

    @Schema(
        title = "Labels that should be added to the flow's execution",
        description = "Any labels passed by the LLM will override those defined here.",
        implementation = Object.class, oneOf = { List.class, Map.class }
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
    @PluginProperty(group = "advanced")
    private final Property<Boolean> inheritLabels = Property.ofValue(false);

    @Schema(
        title = "Schedule the flow execution at a later date",
        description = "If the LLM sets a scheduleDate, it will override the one defined here."
    )
    @PluginProperty(group = "advanced")
    private Property<ZonedDateTime> scheduleDate;

    @Schema(
        title = "Override Kestra API endpoint",
        description = """
            URL used for calls to the Kestra API. When null, renders `{{ kestra.url }}` from configuration; if still empty, defaults to `http://localhost:8080`."""
    )
    @PluginProperty(group = "connection")
    private Property<String> kestraUrl;

    @Schema(
        title = "Select API authentication",
        description = "Use either an API token or HTTP Basic (username/password); do not provide both."
    )
    @PluginProperty(group = "connection")
    private Auth auth;

    @Schema(title = "Override target tenant", description = "Tenant identifier applied to API calls; defaults to the current execution tenant.")
    @PluginProperty(group = "connection")
    private Property<String> tenantId;

    private KestraClient kestraClient(RunContext runContext) throws IllegalVariableEvaluationException {
        String rKestraUrl = runContext.render(kestraUrl).as(String.class)
            .orElseGet(() -> {
                try {
                    return runContext.render("{{ kestra.url }}");
                } catch (IllegalVariableEvaluationException e) {
                    return "http://localhost:8080";
                }
            });
        String normalizedUrl = rKestraUrl.trim().replaceAll("/+$", "");
        var builder = KestraClient.builder();
        builder.url(normalizedUrl);
        if (auth != null) {
            String rApiToken = runContext.render(auth.apiToken).as(String.class).orElse(null);
            if (rApiToken != null) {
                return builder.tokenAuth(rApiToken).build();
            }
            Optional<String> maybeUsername = runContext.render(auth.username).as(String.class);
            Optional<String> maybePassword = runContext.render(auth.password).as(String.class);
            if (maybeUsername.isPresent() && maybePassword.isPresent()) {
                return builder.basicAuth(maybeUsername.get(), maybePassword.get()).build();
            }
            if (runContext.render(auth.auto).as(Boolean.class).orElse(Boolean.TRUE)) {
                Optional<SDK.Auth> autoAuth = runContext.sdk().defaultAuthentication();
                if (autoAuth.isPresent()) {
                    if (autoAuth.get().apiToken().isPresent()) {
                        return builder.tokenAuth(autoAuth.get().apiToken().get()).build();
                    }
                    if (autoAuth.get().username().isPresent() && autoAuth.get().password().isPresent()) {
                        return builder.basicAuth(autoAuth.get().username().get(), autoAuth.get().password().get()).build();
                    }
                }
            }
            throw new IllegalArgumentException("No authentication method provided");
        } else {
            Optional<SDK.Auth> autoAuth = runContext.sdk().defaultAuthentication();
            if (autoAuth.isPresent()) {
                if (autoAuth.get().apiToken().isPresent()) {
                    return builder.tokenAuth(autoAuth.get().apiToken().get()).build();
                }
                if (autoAuth.get().username().isPresent() && autoAuth.get().password().isPresent()) {
                    return builder.basicAuth(autoAuth.get().username().get(), autoAuth.get().password().get()).build();
                }
            }
        }
        return builder.build();
    }

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        boolean hasDefinedFlow = this.namespace != null && this.flowId != null;
        if (this.namespace != null && this.flowId == null) {
            throw new IllegalArgumentException("Flow ID must be specified when you set the namespace");
        }
        if (this.namespace == null && this.flowId != null) {
            throw new IllegalArgumentException("Namespace must be specified when you set the flow ID");
        }

        var rInputs = runContext.render(MapUtils.emptyOnNull(inputs));

        // compute labels
        boolean rInheritedLabels = runContext.render(inheritLabels).as(Boolean.class, additionalVariables).orElse(false);
        List<Label> executionLabels = MapUtils.nestedToFlattenMap(MapUtils.emptyOnNull((Map<String, Object>) runContext.getVariables().get("labels"))).entrySet().stream()
            .map(entry -> new Label(entry.getKey(), entry.getValue().toString()))
            .toList();
        List<Label> rLabels = ListUtils.emptyOnNull(labels).stream().map(throwFunction(label -> new Label(runContext.render(label.key()), runContext.render(label.value())))).toList();

        // resolve tenant id: explicit property overrides, otherwise fall back to current execution's tenant
        String rTenantId = runContext.render(this.tenantId).as(String.class, additionalVariables)
            .orElse(Objects.toString(runContext.flowInfo().tenantId(), ""));

        var client = kestraClient(runContext);

        var jsonSchema = JsonObjectSchema.builder()
            .addProperty(
                "labels", JsonArraySchema.builder().items(
                    JsonObjectSchema.builder()
                        .addStringProperty("key", "The label key.")
                        .addStringProperty("value", "The label value.")
                        .build()
                ).description("The list of labels.")
                    .build()
            )
            .addProperty(
                "scheduleDate", JsonStringSchema.builder()
                    .description(
                        """
                            The scheduled date of the flow. Use it only if the flow needs to be executed later and not immediately.
                            It should be an ISO8601 formatted zoned date time."""
                    )
                    .build()
            );

        if (hasDefinedFlow) {
            var rNamespace = runContext.render(this.namespace).as(String.class, additionalVariables).orElseThrow();
            var rFlowId = runContext.render(this.flowId).as(String.class, additionalVariables).orElseThrow();
            var rRevision = runContext.render(this.revision).as(Integer.class, additionalVariables);

            FlowWithSource flowWithSource;
            try {
                flowWithSource = client.flows().flow(rNamespace, rFlowId, false, false, rTenantId, rRevision.orElse(null));
            } catch (ApiException e) {
                throw new IllegalArgumentException("Unable to find flow '" + rFlowId + "' in namespace '" + rNamespace + "'", e);
            }

            var rDescription = runContext.render(this.description).as(String.class, additionalVariables).orElse(flowWithSource.getDescription());
            if (rDescription == null) {
                throw new IllegalArgumentException(
                    "A description is required either in the tool's description property or in the flow description. "
                        + "Flow " + flowWithSource.getNamespace() + "." + flowWithSource.getId()
                        + " does not have a description, and the tool's description is empty."
                );
            }

            jsonSchema.description(rDescription);
            if (!ListUtils.isEmpty(flowWithSource.getInputs())) {
                jsonSchema.addProperty(
                    "inputs", JsonArraySchema.builder().items(
                        JsonObjectSchema.builder()
                            .addStringProperty("id", "The input id.")
                            .addStringProperty("value", "The input value.")
                            .build()
                    ).description("The list of inputs.")
                        .build()
                );
                // check if there are any mandatory inputs
                if (
                    flowWithSource.getInputs().stream()
                        .anyMatch(input -> Boolean.TRUE.equals(input.getRequired()) && input.getDefaults() == null && !rInputs.containsKey(input.getId()))
                ) {
                    jsonSchema.required("inputs");
                }
            }

            return Map.of(
                ToolSpecification.builder()
                    .name("kestra_flow_" + IdUtils.fromPartsAndSeparator('_', flowWithSource.getNamespace().replace('.', '_'), flowWithSource.getId()))
                    .description(TOOL_DEFINED_DESCRIPTION)
                    .parameters(jsonSchema.build())
                    .build(),
                new KestraDefinedFlowToolExecutor(runContext, client, rTenantId, flowWithSource, rInputs, rInheritedLabels, executionLabels, rLabels)
            );
        } else {
            jsonSchema.description(TOOL_LLM_DESCRIPTION);
            jsonSchema.addProperty("namespace", JsonStringSchema.builder().build());
            jsonSchema.addProperty("flowId", JsonStringSchema.builder().build());
            jsonSchema.addProperty("revision", JsonNumberSchema.builder().build());
            jsonSchema.addProperty(
                "inputs", JsonArraySchema.builder().items(
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
                new KestraLLMFlowToolExecutor(runContext, client, rTenantId, rInputs, rInheritedLabels, executionLabels, rLabels)
            );
        }
    }

    static class KestraDefinedFlowToolExecutor extends AbstractKestraFlowToolExecutor {
        private final FlowWithSource flowWithSource;

        KestraDefinedFlowToolExecutor(RunContext runContext, KestraClient client, String tenantId, FlowWithSource flowWithSource, Map<String, Object> predefinedInputs, boolean inheritedLabels, List<Label> executionLabels,
            List<Label> taskLabels) {
            super(runContext, client, tenantId, predefinedInputs, inheritedLabels, executionLabels, taskLabels);

            this.flowWithSource = flowWithSource;
        }

        @Override
        protected FlowWithSource getFlow(Map<String, Object> parameters) {
            return flowWithSource;
        }
    }

    static class KestraLLMFlowToolExecutor extends AbstractKestraFlowToolExecutor {
        private final KestraClient client;
        private final String tenantId;

        KestraLLMFlowToolExecutor(RunContext runContext, KestraClient client, String tenantId, Map<String, Object> predefinedInputs, boolean inheritedLabels, List<Label> executionLabels, List<Label> taskLabels) {
            super(runContext, client, tenantId, predefinedInputs, inheritedLabels, executionLabels, taskLabels);

            this.client = client;
            this.tenantId = tenantId;
        }

        @Override
        protected FlowWithSource getFlow(Map<String, Object> parameters) {
            var namespace = (String) parameters.get("namespace");
            var flowId = (String) parameters.get("flowId");
            // revision may come back as Double from JSON parsing, so use Number cast
            var revision = Optional.ofNullable(parameters.get("revision"))
                .map(v -> ((Number) v).intValue())
                .orElse(null);
            try {
                return client.flows().flow(namespace, flowId, false, false, tenantId, revision);
            } catch (ApiException e) {
                throw new ToolExecutionException("Flow not found: " + namespace + "." + flowId, e);
            }
        }
    }

    static abstract class AbstractKestraFlowToolExecutor implements ToolExecutor {
        private final RunContext runContext;
        private final KestraClient client;
        private final String tenantId;
        private final Map<String, Object> predefinedInputs;
        private final boolean inheritedLabels;
        private final List<Label> executionLabels;
        private final List<Label> taskLabels;

        AbstractKestraFlowToolExecutor(RunContext runContext, KestraClient client, String tenantId, Map<String, Object> predefinedInputs, boolean inheritedLabels, List<Label> executionLabels, List<Label> taskLabels) {
            this.runContext = runContext;
            this.client = client;
            this.tenantId = tenantId;
            this.predefinedInputs = predefinedInputs;
            this.inheritedLabels = inheritedLabels;
            this.executionLabels = executionLabels;
            this.taskLabels = taskLabels;
        }

        protected abstract FlowWithSource getFlow(Map<String, Object> parameters);

        @Override
        @SuppressWarnings("unchecked")
        public String execute(ToolExecutionRequest toolExecutionRequest, Object memoryId) {
            runContext.logger().debug("Tool execution request: {}", toolExecutionRequest);
            try {
                var flowParameters = JacksonMapper.toMap(toolExecutionRequest.arguments());

                var scheduledDate = Optional.ofNullable((String) flowParameters.get("scheduleDate")).map(d -> ZonedDateTime.parse(d));

                var flowWithSource = getFlow(flowParameters);

                List<Label> newLabels = inheritedLabels ? new ArrayList<>(filterLabels(executionLabels, flowWithSource)) : new ArrayList<>(systemLabels(executionLabels));
                newLabels.addAll(taskLabels);

                // merge LLM provided labels with tool predefined one
                var labels = (List<Map<String, String>>) flowParameters.get("labels");
                var labelList = ListUtils.emptyOnNull(labels).stream()
                    .map(label -> new Label(label.get("key"), label.get("value")))
                    .toList();
                var predefinedLabelsToAdd = newLabels.stream().filter(l1 -> labelList.stream().noneMatch(l2 -> l1.key().equals(l2.key()))).toList();
                var finalLabels = ListUtils.concat(labelList, predefinedLabelsToAdd);

                // build final labels as "key:value" strings for the SDK
                var sdkLabels = finalLabels.stream()
                    .map(l -> l.key() + ":" + l.value())
                    .toList();

                // merge LLM provided inputs with tool predefined one
                var inputs = (List<Map<String, Object>>) flowParameters.get("inputs");
                var inputMap = ListUtils.emptyOnNull(inputs).stream().collect(
                    Collectors.toMap(
                        input -> (String) input.get("id"),
                        input -> input.get("value")
                    )
                );
                var finalInputs = MapUtils.merge(predefinedInputs, inputMap);
                // check mandatory inputs to fail the tool execution instead of triggering a flow that would fail anyway
                ListUtils.emptyOnNull(flowWithSource.getInputs()).forEach(input -> {
                    if (Boolean.TRUE.equals(input.getRequired()) && input.getDefaults() == null && !finalInputs.containsKey(input.getId())) {
                        throw new ToolArgumentsException("You need to provide an input with the id '" + input.getId() + "'.");
                    }
                });

                var response = client.executions().createExecution(
                    flowWithSource.getNamespace(),
                    flowWithSource.getId(),
                    false,
                    tenantId,
                    sdkLabels,
                    flowWithSource.getRevision(),
                    scheduledDate.map(ZonedDateTime::toOffsetDateTime).orElse(null),
                    null,
                    null,
                    new HashMap<>(finalInputs)
                );

                return JacksonMapper.ofJson().writeValueAsString(response);
            } catch (Exception e) {
                throw new ToolExecutionException(e);
            }
        }

        private List<Label> filterLabels(List<Label> labels, FlowWithSource flow) {
            if (ListUtils.isEmpty(flow.getLabels())) {
                return labels;
            }

            // flow.getLabels() returns io.kestra.sdk.model.Label which uses getKey()/getValue()
            return labels.stream()
                .filter(label -> flow.getLabels().stream().noneMatch(flowLabel -> flowLabel.getKey().equals(label.key())))
                .toList();
        }

        private List<Label> systemLabels(List<Label> labels) {
            return labels.stream()
                .filter(label -> label.key().startsWith(Label.SYSTEM_PREFIX))
                .toList();
        }
    }

    @Builder
    @Getter
    public static class Auth {
        @Schema(title = "API token for bearer auth")
        @PluginProperty(group = "connection")
        private Property<String> apiToken;

        @Schema(title = "Username for HTTP Basic auth")
        @PluginProperty(group = "connection")
        private Property<String> username;

        @Schema(title = "Password for HTTP Basic auth")
        @PluginProperty(group = "connection")
        private Property<String> password;

        @Schema(title = "Automatically retrieve credentials from Kestra's configuration if available")
        @Builder.Default
        @PluginProperty(group = "advanced")
        private Property<Boolean> auto = Property.ofValue(Boolean.TRUE);
    }
}
