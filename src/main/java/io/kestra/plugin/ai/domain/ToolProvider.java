package io.kestra.plugin.ai.domain;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.service.tool.DefaultToolExecutor;
import dev.langchain4j.service.tool.ToolExecutor;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.plugins.AdditionalPlugin;
import io.kestra.core.plugins.serdes.PluginDeserializer;
import io.kestra.core.runners.RunContext;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

import static dev.langchain4j.agent.tool.ToolSpecifications.toolSpecificationFrom;
import static dev.langchain4j.service.IllegalConfigurationException.illegalConfiguration;

@Plugin
@SuperBuilder(toBuilder = true)
@Getter
@NoArgsConstructor
// IMPORTANT: The abstract plugin base class must define using the PluginDeserializer,
// AND concrete subclasses must be annotated by @JsonDeserialize() to avoid StackOverflow.
@JsonDeserialize(using = PluginDeserializer.class)
public abstract class ToolProvider extends AdditionalPlugin {
    public abstract Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws IllegalVariableEvaluationException;

    public void close(RunContext runContext) {
        // by default: no-op
    }

    // This method is inspired by ToolServices.tools(List<Object>) from langchain4j
    protected Map<ToolSpecification, ToolExecutor> extract(Object objectWithTool) {
        if (objectWithTool instanceof Class) {
            throw illegalConfiguration("Tool '%s' must be an object, not a class", objectWithTool);
        }

        Map<ToolSpecification, ToolExecutor> tools = new HashMap<>();
        for (Method method : objectWithTool.getClass().getDeclaredMethods()) {
            if (method.isAnnotationPresent(Tool.class)) {
                ToolSpecification toolSpecification = toolSpecificationFrom(method);
                tools.put(toolSpecification, DefaultToolExecutor.builder()
                    .object(objectWithTool)
                    .originalMethod(method)
                    .methodToInvoke(method)
                    .propagateToolExecutionExceptions(true)
                    .wrapToolArgumentsExceptions(true)
                    .build()
                );
            }
        }
        return tools;
    }
}
