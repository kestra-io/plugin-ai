package io.kestra.plugin.ai.tool;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.service.tool.ToolExecutor;
import io.kestra.core.docs.JsonSchemaGenerator;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.tasks.Output;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.DefaultRunContext;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.core.utils.ListUtils;
import io.kestra.core.utils.MapUtils;
import io.kestra.plugin.ai.domain.ToolProvider;
import io.kestra.plugin.ai.tool.internal.JsonObjectSchemaTranslator;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@Plugin(
    examples =  {
        @Example(
                title = "Call a Kestra runnable task as a tool, letting the agent set the `message` property for you",
            full = true,
            code = {
                """
                    id: call_a_kestra_task
                    namespace: company.ai

                    tasks:
                      - id: agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-2.5-flash
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.KestraTask
                            tasks:
                              - id: log
                                type: io.kestra.plugin.core.log.Log
                                message: "..." # This is a placeholder; the agent will fill it.
                        prompt: "Log the following message: 'Hello World!'"
                """
            }
        ),
    }
)
@JsonDeserialize
@Schema(
    title = "Expose Kestra runnable tasks as tools",
    description = """
        Creates one tool per runnable task named `kestra_task_<taskId>`. Properties you set stay fixed; set a required property to `...` to force the agent to supply it, and unset optionals may be filled by the agent. `anyOf` schemas are flattened to a single branch because many models do not support `anyOf`; the generated schema appears in debug logs."""
)
public class KestraTask extends ToolProvider {
    // This placeholder would be used in the flow definition to denote a property that the LLM must set.
    private static final String LLM_PLACEHOLDER = "...";

    // Tool description, it could be fine-tuned if needed
    private static final String TOOL_DESCRIPTION = "This tool allows you to call a Kestra task. A Kestra task will respond with its output, which is a map of key/value pairs.";

    @Schema(title = "List of Kestra runnable tasks")
    @NotNull
    private List<Task> tasks;

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException {
        var defaultRunContext = (DefaultRunContext) runContext;
        var jsonSchemaGenerator = defaultRunContext.getApplicationContext().getBean(JsonSchemaGenerator.class);

        Map<ToolSpecification, ToolExecutor> tools = new HashMap<>();
        for (Task task : tasks) {
            // First, check that the task is a runnable task.
            if (!(task instanceof RunnableTask)) {
                throw new IllegalArgumentException("The KestraTask tool is only capable of calling runnable tasks but '" + task.getId() + "' is not a runnable task.");
            }

            var schemaAnnotation = Optional.ofNullable(task.getClass().getAnnotation(Schema.class));
            if (schemaAnnotation.isEmpty()) {
                runContext.logger().warn("The task {} has no description, so the LLM may not understand what is its purpose; you may need to explicitly describe it in the prompt.", task.getId());
            }
            var description = schemaAnnotation.map(s -> s.title()).orElse(null);
            var schema = jsonSchemaGenerator.properties(Task.class, task.getClass());
            var taskProperties = JacksonMapper.toMap(task);

            // we will remove from the schema what's already set as taskProperties
            removeAlreadySet(schema, taskProperties);
            // then transform the schema as a Langchain4J schema
            var parameters = JsonObjectSchemaTranslator.fromOpenAPISchema(schema, description);
            runContext.logger().debug("Generated JSON schema:\n{}", parameters);

            var toolSpecification = ToolSpecification.builder()
                .name("kestra_task_" + task.getId())
                .description(TOOL_DESCRIPTION)
                .parameters(parameters)
                .build();
            runContext.logger().debug("Tool specification: {}", toolSpecification);
            var toolExecutor = new KestraTaskToolExecutor((RunnableTask<?>) task, runContext);
            tools.put(toolSpecification, toolExecutor);
        }

        return tools;
    }

    @SuppressWarnings("unchecked")
    private void removeAlreadySet(Map<String, Object> schema, Map<String, Object> taskProperties) {
        var properties = (Map<String, Object>) schema.get("properties");
        var required = (List<String>) schema.get("required");

        properties = MapUtils.emptyOnNull(properties).entrySet().stream()
            .filter(entry -> !taskProperties.containsKey(entry.getKey()) || taskProperties.get(entry.getKey()).equals("...")) // "..." is the placeholder for the LLM agent
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        schema.put("properties", properties);

        required = ListUtils.emptyOnNull(required).stream()
            .filter(entry -> !taskProperties.containsKey(entry) || taskProperties.get(entry).equals("...")) // "..." is the placeholder for the LLM agent
            .toList();
        schema.put("required", required);
    }

    static class KestraTaskToolExecutor implements ToolExecutor {
        private final RunnableTask<?> task;
        private final RunContext runContext;

        KestraTaskToolExecutor(RunnableTask<?> task,  RunContext runContext) {
            this.task = task;
            this.runContext = runContext;
        }

        @Override
        public String execute(ToolExecutionRequest toolExecutionRequest, Object memoryId) {
            runContext.logger().debug("Tool execution request: {}", toolExecutionRequest);
            try {
                // merge task properties with tool execution arguments
                Map<String, Object> taskProperties = JacksonMapper.toMap(task);
                Map<String, Object> arguments = JacksonMapper.toMap(toolExecutionRequest.arguments());
                RunnableTask<?> runnable = JacksonMapper.ofJson().convertValue(MapUtils.merge(taskProperties, arguments), task.getClass());
                Output output = runnable.run(runContext);
                if (output != null) {
                    Map<String, Object> outputMap = output.toMap();
                    if (!MapUtils.isEmpty(outputMap)) {
                        return JacksonMapper.ofJson().writeValueAsString(outputMap);
                    }
                }

                // we return Success here so when a task has no output but reply successfully, the LLM knows it and didn't re-call the task
                return "Success";
            } catch (Exception e) {
                // TODO we may instead send the error to the LLM so it can decide to retry it or not
                throw new ToolExecutionException(e);
            }
        }
    }
}
