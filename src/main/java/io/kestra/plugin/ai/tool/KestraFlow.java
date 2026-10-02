package io.kestra.plugin.ai.tool;

import java.time.OffsetDateTime;
import java.time.ZonedDateTime;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import com.fasterxml.jackson.core.type.TypeReference;
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
import io.kestra.core.tenant.TenantService;
import io.kestra.core.utils.IdUtils;
import io.kestra.core.utils.ListUtils;
import io.kestra.core.utils.MapUtils;
import io.kestra.core.validations.NoSystemLabelValidation;
import io.kestra.plugin.ai.domain.ToolProvider;
import io.kestra.sdk.KestraClient;
import io.kestra.sdk.api.ExecutionsApi;
import io.kestra.sdk.internal.ApiClient;
import io.kestra.sdk.internal.ApiException;
import io.kestra.sdk.internal.Pair;
import io.kestra.sdk.model.ExecutionControllerExecutionResponse;
import io.kestra.sdk.model.ExecutionKind;
import io.kestra.sdk.model.FlowWithSource;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.exception.LangChain4jException;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonNumberSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import dev.langchain4j.service.tool.ToolExecutor;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
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
                          modelName: gemini-3.5-flash-lite
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: business-automation
                            description: Business Automation
                            auth:
                              apiToken: "{{ secret('KESTRA_API_TOKEN') }}"

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: business-processes
                            description: Business Processes
                            auth:
                              apiToken: "{{ secret('KESTRA_API_TOKEN') }}"

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: data-engineering-pipeline
                            description: Data Engineering Pipeline
                            auth:
                              apiToken: "{{ secret('KESTRA_API_TOKEN') }}"

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: dwh-and-analytics
                            description: Data Warehouse and Analytics
                            auth:
                              apiToken: "{{ secret('KESTRA_API_TOKEN') }}"

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: file-processing
                            description: File Processing
                            auth:
                              apiToken: "{{ secret('KESTRA_API_TOKEN') }}"

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: hello-world
                            description: Hello World
                            auth:
                              apiToken: "{{ secret('KESTRA_API_TOKEN') }}"

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: infrastructure-automation
                            description: Infrastructure Automation
                            auth:
                              apiToken: "{{ secret('KESTRA_API_TOKEN') }}"

                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            namespace: tutorial
                            flowId: microservices-and-apis
                            description: Microservices and APIs
                            auth:
                              apiToken: "{{ secret('KESTRA_API_TOKEN') }}\""""
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
                          modelName: gemini-3.5-flash-lite
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            auth:
                              apiToken: "{{ secret('KESTRA_API_TOKEN') }}\""""
            }
        ),
        @Example(
            title = "Limit an agent to explicitly allowed flows",
            full = true,
            code = {
                """
                    id: agent_calling_allowed_flows
                    namespace: company.ai

                    tasks:
                      - id: agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        prompt: Execute the hello-world flow in the tutorial namespace.
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-3.5-flash-lite
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.KestraFlow
                            allowedFlows:
                              - namespace: tutorial
                                flowId: hello-world
                            auth:
                              apiToken: "{{ secret('KESTRA_API_TOKEN') }}"
                    """
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

    private static final String DEFAULT_URL = "http://localhost:8080";
    private static final String URL_TEMPLATE = "{{ kestra.url }}";

    @Schema(
        title = "Tool description",
        description = "Natural-language summary of what the called flow does, which the LLM uses to decide whether to call it. Not set by default: the target flow's own description is used, so this property is only needed when that flow has none, or when the flow is chosen dynamically through `allowedFlows`.",
        example = "Sends a Slack notification to the on-call channel."
    )
    @PluginProperty(group = "advanced")
    private Property<String> description;

    @Schema(
        title = "Flow namespace",
        description = "Namespace of the flow to execute. Not set by default, in which case the LLM chooses the namespace, constrained by `allowedFlows` when it is configured.",
        example = "company.team"
    )
    @PluginProperty(group = "connection")
    private Property<String> namespace;

    @Schema(
        title = "Flow ID",
        description = "Identifier of the flow to execute. Not set by default, in which case the LLM chooses the flow, constrained by `allowedFlows` when it is configured.",
        example = "send_notification"
    )
    @PluginProperty(group = "advanced")
    private Property<String> flowId;

    @Schema(
        title = "Allowed flows",
        description = "Allowlist of exact namespace and flow ID pairs the tool may execute. Not set by default, meaning no restriction. When set, it must be non-empty and every entry must resolve to a non-blank namespace and flow ID; the permitted pairs are exposed to the model and any selection outside the list is rejected before the API is called. The restriction applies even when `namespace` and `flowId` are predefined on the tool.",
        example = "[{namespace: \"company.team\", flowId: \"send_notification\"}]"
    )
    @Valid
    @PluginProperty(group = "connection")
    private List<AllowedFlow> allowedFlows;

    @Schema(
        title = "Flow revision",
        description = "Specific revision of the flow to execute. Not set by default, in which case the latest revision runs.",
        example = "3"
    )
    @PluginProperty(group = "advanced")
    private Property<Integer> revision;

    @Schema(
        title = "Flow execution inputs",
        description = "Input values passed to the triggered execution. Any input the LLM supplies overrides the value defined here. Not set by default.",
        example = "{channel: \"#on-call\", severity: \"high\"}"
    )
    @PluginProperty(dynamic = true, group = "advanced")
    private Map<String, Object> inputs;

    @Schema(
        title = "Flow execution labels",
        description = "Labels added to the triggered execution. Any label the LLM supplies overrides the value defined here. Not set by default.",
        example = "{triggeredBy: \"ai-agent\"}",
        implementation = Object.class, oneOf = { List.class, Map.class }
    )
    @PluginProperty(dynamic = true, group = "advanced")
    @JsonSerialize(using = ListOrMapOfLabelSerializer.class)
    @JsonDeserialize(using = ListOrMapOfLabelDeserializer.class)
    private List<@NoSystemLabelValidation Label> labels;

    @Builder.Default
    @Schema(
        title = "Inherit labels from the calling execution",
        description = "If `true`, the triggered execution inherits all labels from the agent's own execution. Defaults to `false`. Any label the LLM supplies still takes precedence.",
        example = "true"
    )
    @PluginProperty(group = "advanced")
    private final Property<Boolean> inheritLabels = Property.ofValue(false);

    @Schema(
        title = "Scheduled execution date",
        description = "Date and time at which the execution should start, rather than immediately. Not set by default (immediate execution). A `scheduleDate` supplied by the LLM overrides this value.",
        example = "2026-01-01T09:00:00Z"
    )
    @PluginProperty(group = "advanced")
    private Property<ZonedDateTime> scheduleDate;

    @Schema(
        title = "Kestra API endpoint",
        description = "Base URL used for calls to the Kestra API. Not set by default, in which case `{{ kestra.url }}` is rendered from configuration, falling back to `http://localhost:8080`.",
        example = "https://kestra.internal:8080"
    )
    @PluginProperty(group = "connection")
    private Property<String> kestraUrl;

    @Schema(
        title = "API authentication",
        description = "Credentials used to call the Kestra API: either an API token or HTTP Basic username/password, never both. Not set by default, in which case credentials are taken from Kestra's own configuration.",
        example = "{apiToken: \"{{ secret('KESTRA_API_TOKEN') }}\"}"
    )
    @PluginProperty(group = "connection")
    private Auth auth;

    @Schema(
        title = "Target tenant",
        description = "Tenant the API calls are made against. Defaults to the tenant of the current execution.",
        example = "main"
    )
    @PluginProperty(group = "connection")
    private Property<String> tenantId;

    private Optional<KestraClient> tryAutoAuth(KestraClient.KestraClientBuilder builder, RunContext runContext) {
        SDK sdk = runContext.sdk();
        if (sdk == null) return Optional.empty();
        Optional<SDK.Auth> autoAuth = sdk.defaultAuthentication();
        if (autoAuth.isPresent()) {
            if (autoAuth.get().apiToken().isPresent()) {
                return Optional.of(builder.tokenAuth(autoAuth.get().apiToken().get()).build());
            }
            if (autoAuth.get().username().isPresent() && autoAuth.get().password().isPresent()) {
                return Optional.of(builder.basicAuth(autoAuth.get().username().get(), autoAuth.get().password().get()).build());
            }
        }
        return Optional.empty();
    }

    private String resolveUrl(RunContext runContext) throws IllegalVariableEvaluationException {
        String rUrl = runContext.render(kestraUrl).as(String.class)
            .filter(Predicate.not(String::isBlank))
            .orElseGet(() -> configuredUrl(runContext));

        return rUrl.trim().replaceAll("/+$", "");
    }

    private static String configuredUrl(RunContext runContext) {
        try {
            String rUrl = runContext.render(URL_TEMPLATE);
            return rUrl == null || rUrl.isBlank() ? DEFAULT_URL : rUrl;
        } catch (IllegalVariableEvaluationException e) {
            return DEFAULT_URL;
        }
    }

    private static IllegalArgumentException noAuthentication() {
        return new IllegalArgumentException(
            "No authentication method provided. Set the `auth` property of the tool, or configure a default one with the `kestra.tasks.sdk.authentication` properties. If this API requires no authentication, set `auth.auto` to false and leave the credentials unset."
        );
    }

    private KestraClient kestraClient(RunContext runContext) throws IllegalVariableEvaluationException {
        var builder = KestraClient.builder();
        builder.url(resolveUrl(runContext));

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
                return tryAutoAuth(builder, runContext).orElseThrow(KestraFlow::noAuthentication);
            }

            return builder.noAuth().build();
        }

        return tryAutoAuth(builder, runContext).orElseThrow(KestraFlow::noAuthentication);
    }

    /** A 401 or a 403 from the API is a credentials problem, and reporting it as a missing flow sends the user looking in the wrong place. */
    private static String apiFailureMessage(ApiException e, String namespace, String flowId) {
        return switch (e.getCode()) {
            case 401 -> "Authentication failed when calling the Kestra API for the flow '%s' in the namespace '%s'. Check the credentials set in the `auth` property of the tool.".formatted(flowId, namespace);
            case 403 -> "Not authorized to access the flow '%s' in the namespace '%s'. Check the permissions of the credentials set in the `auth` property of the tool.".formatted(flowId, namespace);
            case 404 -> "Unable to find the flow '%s' in the namespace '%s'.".formatted(flowId, namespace);
            case 0 -> "The Kestra API could not be reached for the flow '%s' in the namespace '%s': %s".formatted(flowId, namespace, e.getMessage());
            default -> "The Kestra API returned the status %d for the flow '%s' in the namespace '%s': %s".formatted(e.getCode(), flowId, namespace, e.getMessage());
        };
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

        var rAllowedFlows = resolveAllowedFlows(runContext, additionalVariables);
        var rInputs = runContext.render(MapUtils.emptyOnNull(inputs));

        // compute labels
        boolean rInheritedLabels = runContext.render(inheritLabels).as(Boolean.class, additionalVariables).orElse(false);
        List<Label> executionLabels = MapUtils.nestedToFlattenMap(MapUtils.emptyOnNull((Map<String, Object>) runContext.getVariables().get("labels"))).entrySet().stream()
            .map(entry -> new Label(entry.getKey(), entry.getValue().toString()))
            .toList();
        List<Label> rLabels = ListUtils.emptyOnNull(labels).stream().map(throwFunction(label -> new Label(runContext.render(label.key()), runContext.render(label.value())))).toList();

        // resolve tenant id: explicit property overrides, otherwise fall back to current execution's tenant
        String rTenantId = runContext.render(this.tenantId).as(String.class, additionalVariables)
            .filter(Predicate.not(String::isBlank))
            .or(() -> Optional.ofNullable(runContext.flowInfo().tenantId()))
            .orElse(TenantService.MAIN_TENANT);

        var client = kestraClient(runContext);

        var inputsSchema = JsonArraySchema.builder().items(
            JsonObjectSchema.builder()
                .addStringProperty("id", "The input id.")
                .addStringProperty("value", "The input value.")
                .build()
        ).description("The list of inputs.").build();

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

            if (rAllowedFlows != null && !rAllowedFlows.contains(new FlowIdentifier(rNamespace, rFlowId))) {
                throw new IllegalArgumentException(
                    "The predefined flow '%s' in namespace '%s' is not in allowedFlows. Add the pair to allowedFlows or select an allowed flow.".formatted(rFlowId, rNamespace)
                );
            }

            FlowWithSource flowWithSource;
            try {
                flowWithSource = client.flows().flow(rNamespace, rFlowId, rTenantId, false, rRevision.orElse(null), false);
            } catch (ApiException e) {
                throw new IllegalArgumentException(apiFailureMessage(e, rNamespace, rFlowId), e);
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
                jsonSchema.addProperty("inputs", inputsSchema);
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
            var toolDescription = TOOL_LLM_DESCRIPTION;
            var namespaceDescription = "Namespace of an existing Kestra flow. Must be paired with that flow's ID; do not invent a namespace.";
            var flowIdDescription = "ID of an existing Kestra flow in the selected namespace. Do not invent a flow ID.";
            if (rAllowedFlows == null) {
                jsonSchema.addProperty("namespace", JsonStringSchema.builder().description(namespaceDescription).build());
                jsonSchema.addProperty("flowId", JsonStringSchema.builder().description(flowIdDescription).build());
            } else {
                toolDescription += "\nSelect only one of these exact (namespace, flowId) pairs: " + rAllowedFlows.stream()
                    .map(flow -> "(%s, %s)".formatted(flow.namespace(), flow.flowId()))
                    .collect(Collectors.joining(", ")) + ". Do not combine values from different pairs.";
                jsonSchema.addProperty(
                    "namespace", JsonEnumSchema.builder()
                        .description(namespaceDescription)
                        .enumValues(rAllowedFlows.stream().map(FlowIdentifier::namespace).distinct().toList())
                        .build()
                );
                jsonSchema.addProperty(
                    "flowId", JsonEnumSchema.builder()
                        .description(flowIdDescription)
                        .enumValues(rAllowedFlows.stream().map(FlowIdentifier::flowId).distinct().toList())
                        .build()
                );
            }
            jsonSchema.description(toolDescription);
            jsonSchema.addProperty("revision", JsonNumberSchema.builder().build());
            jsonSchema.addProperty("inputs", inputsSchema);
            jsonSchema.required("namespace", "flowId");

            return Map.of(
                ToolSpecification.builder()
                    .name("kestra_flow")
                    .description(toolDescription)
                    .parameters(jsonSchema.build())
                    .build(),
                new KestraLLMFlowToolExecutor(runContext, client, rTenantId, rInputs, rInheritedLabels, executionLabels, rLabels, rAllowedFlows)
            );
        }
    }

    private List<FlowIdentifier> resolveAllowedFlows(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        if (allowedFlows == null) {
            return null;
        }
        if (allowedFlows.isEmpty()) {
            throw new IllegalArgumentException("allowedFlows must contain at least one flow when configured. Add an allowed namespace and flowId pair.");
        }
        var rAllowedFlows = new ArrayList<FlowIdentifier>();
        for (var flow : allowedFlows) {
            if (flow == null) {
                throw new IllegalArgumentException("Each allowedFlows entry must specify a namespace and flowId; null entries are not allowed.");
            }
            var rNamespace = runContext.render(flow.namespace).as(String.class, additionalVariables)
                .filter(Predicate.not(String::isBlank))
                .orElseThrow(() -> new IllegalArgumentException("Each allowedFlows entry must specify a nonblank namespace."));
            var rFlowId = runContext.render(flow.flowId).as(String.class, additionalVariables)
                .filter(Predicate.not(String::isBlank))
                .orElseThrow(() -> new IllegalArgumentException("Each allowedFlows entry must specify a nonblank flowId."));
            rAllowedFlows.add(new FlowIdentifier(rNamespace, rFlowId));
        }
        return List.copyOf(rAllowedFlows);
    }

    private record FlowIdentifier(String namespace, String flowId) {
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
        private final List<FlowIdentifier> allowedFlows;

        KestraLLMFlowToolExecutor(RunContext runContext, KestraClient client, String tenantId, Map<String, Object> predefinedInputs, boolean inheritedLabels, List<Label> executionLabels,
            List<Label> taskLabels, List<FlowIdentifier> allowedFlows) {
            super(runContext, client, tenantId, predefinedInputs, inheritedLabels, executionLabels, taskLabels);
            this.allowedFlows = allowedFlows;
        }

        @Override
        protected FlowWithSource getFlow(Map<String, Object> parameters) {
            var namespace = (String) parameters.get("namespace");
            var flowId = (String) parameters.get("flowId");
            if (allowedFlows != null && !allowedFlows.contains(new FlowIdentifier(namespace, flowId))) {
                throw new ToolArgumentsException(
                    "The flow '%s' in namespace '%s' is not in allowedFlows. Select an exact namespace and flowId pair from the tool description.".formatted(flowId, namespace)
                );
            }
            // revision may come back as Double from JSON parsing, so use Number cast
            var revision = Optional.ofNullable(parameters.get("revision"))
                .map(v -> ((Number) v).intValue())
                .orElse(null);
            try {
                return client.flows().flow(namespace, flowId, tenantId, false, revision, false);
            } catch (ApiException e) {
                // langchain4j reports the root cause's message to the agent, so chaining the ApiException would hide this one
                runContext.logger().warn("Calling the Kestra API failed with the status {}.", e.getCode(), e);
                throw new ToolExecutionException(apiFailureMessage(e, namespace, flowId));
            }
        }
    }

    static abstract class AbstractKestraFlowToolExecutor implements ToolExecutor {
        protected final RunContext runContext;
        protected final KestraClient client;
        protected final String tenantId;
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

                var executionsApi = new ExecutionsApiWithInputs(client.executions().getApiClient());
                ExecutionControllerExecutionResponse response;
                try {
                    response = executionsApi.createExecutionWithInputs(
                        tenantId,
                        flowWithSource.getNamespace(),
                        flowWithSource.getId(),
                        sdkLabels,
                        false,
                        flowWithSource.getRevision(),
                        scheduledDate.map(ZonedDateTime::toOffsetDateTime).orElse(null),
                        null,
                        null,
                        new HashMap<>(finalInputs)
                    );
                } catch (ApiException e) {
                    // langchain4j reports the root cause's message to the agent, so chaining the ApiException would hide this one
                    runContext.logger().warn("Calling the Kestra API failed with the status {}.", e.getCode(), e);
                    throw new ToolExecutionException(apiFailureMessage(e, flowWithSource.getNamespace(), flowWithSource.getId()));
                }

                return JacksonMapper.ofJson().writeValueAsString(response);
            } catch (LangChain4jException e) {
                throw e;
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

    private static final class ExecutionsApiWithInputs extends ExecutionsApi {
        private static final String MULTIPART = "multipart/form-data";

        ExecutionsApiWithInputs(ApiClient apiClient) {
            super(apiClient);
        }

        ExecutionControllerExecutionResponse createExecutionWithInputs(
                String tenant, String namespace, String id,
                List<String> labels, Boolean wait, Integer revision,
                OffsetDateTime scheduleDate, String breakpoints, ExecutionKind kind,
                Map<String, Object> inputs) throws ApiException {
            List<Pair> multiLabels = labels == null || labels.isEmpty()
                ? Collections.emptyList()
                : apiClient.parameterToPairs("multi", "labels", labels);
            return invoke("POST",
                tenantPath(tenant, "executions", namespace, id),
                null,
                queryParams("wait", wait, "revision", revision,
                    "scheduleDate", scheduleDate, "breakpoints", breakpoints, "kind", kind),
                multiLabels,
                JSON, MULTIPART,
                inputs != null ? inputs : new HashMap<>(),
                new TypeReference<ExecutionControllerExecutionResponse>() {});
        }
    }

    @Builder
    @Getter
    @Schema(title = "An allowed flow")
    public static class AllowedFlow {
        @Schema(
            title = "Allowed flow namespace",
            description = "Namespace of a flow the tool is permitted to execute. No default: this property is required on each allowlist entry.",
            example = "company.team"
        )
        @NotNull
        @PluginProperty(group = "main")
        private Property<String> namespace;

        @Schema(
            title = "Allowed flow ID",
            description = "Identifier of a flow the tool is permitted to execute. No default: this property is required on each allowlist entry.",
            example = "send_notification"
        )
        @NotNull
        @PluginProperty(group = "main")
        private Property<String> flowId;
    }

    @Builder
    @Getter
    public static class Auth {
        @Schema(
            title = "API token",
            description = "Bearer token authenticating calls to the Kestra API. Store it as a Kestra secret rather than inline. Mutually exclusive with `username`/`password`.",
            example = "{{ secret('KESTRA_API_TOKEN') }}"
        )
        @PluginProperty(secret = true, group = "connection")
        private Property<String> apiToken;

        @Schema(
            title = "HTTP Basic username",
            description = "User authenticating against the Kestra API with HTTP Basic. Must be paired with `password` and is mutually exclusive with `apiToken`.",
            example = "admin@kestra.io"
        )
        @PluginProperty(group = "connection")
        private Property<String> username;

        @Schema(
            title = "HTTP Basic password",
            description = "Password paired with `username` for HTTP Basic authentication. Store it as a Kestra secret rather than inline. Mutually exclusive with `apiToken`.",
            example = "{{ secret('KESTRA_PASSWORD') }}"
        )
        @PluginProperty(secret = true, group = "connection")
        private Property<String> password;

        @Schema(
            title = "Auto-retrieve credentials",
            description = "If `true`, missing credentials are taken from Kestra's own configuration when available. Defaults to `true`. Set it to `false`, with no credentials, to call a Kestra API that requires no authentication.",
            example = "false"
        )
        @Builder.Default
        @PluginProperty(group = "advanced")
        private Property<Boolean> auto = Property.ofValue(Boolean.TRUE);
    }
}
