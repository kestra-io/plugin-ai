package io.kestra.plugin.ai.tool;

import java.io.InputStream;
import java.net.URI;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.core.utils.ListUtils;
import io.kestra.plugin.ai.domain.ToolProvider;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.skills.DefaultSkill;
import dev.langchain4j.skills.DefaultSkillResource;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Getter
@SuperBuilder
@NoArgsConstructor
@Plugin(
    examples = {
        @Example(
            title = "Use skills to provide structured instructions to an AI agent",
            full = true,
            code = {
                """
                    id: agent_with_skills
                    namespace: company.ai

                    tasks:
                      - id: agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        prompt: Translate the following text to French - "Hello, how are you today?"
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-2.5-flash
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.Skill
                            skills:
                              - name: translation_expert
                                description: Expert translator for multiple languages
                                content: |
                                  You are an expert translator. When translating text:
                                  1. Preserve the original meaning and tone
                                  2. Use natural phrasing in the target language
                                  3. Keep proper nouns unchanged"""
            }
        ),
    }
)
@JsonDeserialize
@Schema(
    title = "Provide skills to an AI agent",
    description = """
        Exposes langchain4j skills as tools for an AI agent. Skills are structured instructions
        that the agent can activate on demand. Each skill has a name, description, and content
        that gets returned when the agent activates it. Skills can also include resources
        that the agent can read separately."""
)
public class Skill extends ToolProvider {

    @Schema(
        title = "List of skill definitions",
        description = """
            Each skill defines a set of structured instructions that the agent can activate.
            A skill must have a name, description, and either inline content or a content URI
            pointing to Kestra internal storage."""
    )
    @NotNull
    @PluginProperty
    private List<SkillDefinition> skills;

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws Exception {
        var skillList = ListUtils.emptyOnNull(skills).stream()
            .map(def -> buildSkill(runContext, additionalVariables, def))
            .toList();

        if (skillList.isEmpty()) {
            throw new IllegalArgumentException("At least one skill must be defined");
        }

        // Build a map for quick lookup by name
        var skillMap = new HashMap<String, dev.langchain4j.skills.Skill>();
        for (var skill : skillList) {
            skillMap.put(skill.name(), skill);
        }

        // Build a concise skill catalog for the tool description
        var catalogBuilder = new StringBuilder();
        for (var skill : skillList) {
            catalogBuilder.append("- ").append(skill.name()).append(": ").append(skill.description()).append("\n");
        }

        var tools = new HashMap<ToolSpecification, ToolExecutor>();

        // activate_skill tool
        var activateSpec = ToolSpecification.builder()
            .name("activate_skill")
            .description("Activates a skill and returns its content. Available skills:\n" + catalogBuilder)
            .parameters(
                JsonObjectSchema.builder()
                    .addProperty("skill_name", JsonStringSchema.builder()
                        .description("The name of the skill to activate")
                        .build())
                    .required("skill_name")
                    .build()
            )
            .build();
        tools.put(activateSpec, new ActivateSkillExecutor(runContext, skillMap));

        // read_skill_resource tool (only when at least one skill has resources)
        var hasResources = skillList.stream()
            .anyMatch(skill -> !ListUtils.isEmpty(skill.resources()));
        if (hasResources) {
            var readResourceSpec = ToolSpecification.builder()
                .name("read_skill_resource")
                .description("Reads a resource file attached to a skill. Provide the skill name and the relative path of the resource.")
                .parameters(
                    JsonObjectSchema.builder()
                        .addProperty("skill_name", JsonStringSchema.builder()
                            .description("The name of the skill that owns the resource")
                            .build())
                        .addProperty("relative_path", JsonStringSchema.builder()
                            .description("The relative path of the resource within the skill")
                            .build())
                        .required("skill_name", "relative_path")
                        .build()
                )
                .build();
            tools.put(readResourceSpec, new ReadResourceExecutor(runContext, skillMap));
        }

        return tools;
    }

    private dev.langchain4j.skills.Skill buildSkill(RunContext runContext, Map<String, Object> additionalVariables, SkillDefinition def) {
        try {
            var rName = runContext.render(def.getName()).as(String.class, additionalVariables).orElseThrow();
            var rDescription = runContext.render(def.getDescription()).as(String.class, additionalVariables).orElseThrow();
            var rContent = def.getContent() != null
                ? runContext.render(def.getContent()).as(String.class, additionalVariables).orElse(null)
                : null;
            var rContentUri = def.getContentUri() != null
                ? runContext.render(def.getContentUri()).as(String.class, additionalVariables).orElse(null)
                : null;

            if (rContent == null && rContentUri == null) {
                throw new IllegalArgumentException("Skill '" + rName + "' must have either 'content' or 'contentUri' set");
            }
            if (rContent != null && rContentUri != null) {
                throw new IllegalArgumentException("Skill '" + rName + "' must have either 'content' or 'contentUri' set, not both");
            }

            var resolvedContent = rContent;
            if (rContentUri != null) {
                try (InputStream file = runContext.storage().getFile(URI.create(rContentUri))) {
                    resolvedContent = new String(file.readAllBytes());
                }
            }

            var resources = ListUtils.emptyOnNull(def.getResources()).stream()
                .map(resourceDef -> {
                    try {
                        var rRelativePath = runContext.render(resourceDef.getRelativePath()).as(String.class, additionalVariables).orElseThrow();
                        var rResourceContent = runContext.render(resourceDef.getContent()).as(String.class, additionalVariables).orElseThrow();
                        return (dev.langchain4j.skills.SkillResource) DefaultSkillResource.builder()
                            .relativePath(rRelativePath)
                            .content(rResourceContent)
                            .build();
                    } catch (IllegalVariableEvaluationException e) {
                        throw new IllegalStateException("Failed to render skill resource properties", e);
                    }
                })
                .toList();

            return DefaultSkill.builder()
                .name(rName)
                .description(rDescription)
                .content(resolvedContent)
                .resources(resources)
                .build();
        } catch (IllegalVariableEvaluationException e) {
            throw new IllegalStateException("Failed to render skill properties", e);
        } catch (IllegalArgumentException | IllegalStateException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("Failed to build skill", e);
        }
    }

    private static class ActivateSkillExecutor implements ToolExecutor {
        private final RunContext runContext;
        private final Map<String, dev.langchain4j.skills.Skill> skillMap;

        ActivateSkillExecutor(RunContext runContext, Map<String, dev.langchain4j.skills.Skill> skillMap) {
            this.runContext = runContext;
            this.skillMap = skillMap;
        }

        @Override
        public String execute(ToolExecutionRequest request, Object memoryId) {
            runContext.logger().debug("activate_skill request: {}", request);
            try {
                var parameters = JacksonMapper.toMap(request.arguments());
                var skillName = (String) parameters.get("skill_name");
                var skill = skillMap.get(skillName);
                if (skill == null) {
                    throw new IllegalArgumentException("Skill not found: '" + skillName + "'. Available skills: " + skillMap.keySet());
                }
                runContext.logger().info("Activated skill: {}", skillName);
                return skill.content();
            } catch (Exception e) {
                throw new ToolExecutionException(e);
            }
        }
    }

    private static class ReadResourceExecutor implements ToolExecutor {
        private final RunContext runContext;
        private final Map<String, dev.langchain4j.skills.Skill> skillMap;

        ReadResourceExecutor(RunContext runContext, Map<String, dev.langchain4j.skills.Skill> skillMap) {
            this.runContext = runContext;
            this.skillMap = skillMap;
        }

        @Override
        public String execute(ToolExecutionRequest request, Object memoryId) {
            runContext.logger().debug("read_skill_resource request: {}", request);
            try {
                var parameters = JacksonMapper.toMap(request.arguments());
                var skillName = (String) parameters.get("skill_name");
                var relativePath = (String) parameters.get("relative_path");

                var skill = skillMap.get(skillName);
                if (skill == null) {
                    throw new IllegalArgumentException("Skill not found: '" + skillName + "'. Available skills: " + skillMap.keySet());
                }

                var resource = ListUtils.emptyOnNull(skill.resources()).stream()
                    .filter(r -> r.relativePath().equals(relativePath))
                    .findFirst()
                    .orElseThrow(() -> new IllegalArgumentException(
                        "Resource not found: '" + relativePath + "' in skill '" + skillName + "'"
                    ));

                runContext.logger().info("Read resource '{}' from skill '{}'", relativePath, skillName);
                return resource.content();
            } catch (Exception e) {
                throw new ToolExecutionException(e);
            }
        }
    }

    @Getter
    @Builder
    @Schema(title = "A skill definition")
    public static class SkillDefinition {
        @Schema(title = "Name of the skill")
        @NotNull
        private Property<String> name;

        @Schema(title = "Description of the skill used by the LLM to decide when to activate it")
        @NotNull
        private Property<String> description;

        @Schema(
            title = "Inline content of the skill",
            description = "Mutually exclusive with 'contentUri'. At least one of 'content' or 'contentUri' must be set."
        )
        private Property<String> content;

        @Schema(
            title = "URI to the skill content in Kestra internal storage",
            description = "Mutually exclusive with 'content'. At least one of 'content' or 'contentUri' must be set."
        )
        @PluginProperty(internalStorageURI = true)
        private Property<String> contentUri;

        @Schema(
            title = "Additional resources attached to this skill",
            description = "Resources the agent can read separately using the 'read_skill_resource' tool."
        )
        @PluginProperty
        private List<ResourceDefinition> resources;
    }

    @Getter
    @Builder
    @Schema(title = "A skill resource definition")
    public static class ResourceDefinition {
        @Schema(title = "Relative path of the resource within the skill")
        @NotNull
        private Property<String> relativePath;

        @Schema(title = "Content of the resource")
        @NotNull
        private Property<String> content;
    }
}
