package io.kestra.plugin.ai.tool;

import java.io.InputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.ListUtils;
import io.kestra.plugin.ai.domain.ToolProvider;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.invocation.InvocationContext;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.service.tool.ToolProviderRequest;
import dev.langchain4j.skills.DefaultSkill;
import dev.langchain4j.skills.DefaultSkillResource;
import dev.langchain4j.skills.Skills;
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
        @Example(
            title = "Load skill content from Kestra internal storage",
            full = true,
            code = {
                """
                    id: agent_with_skill_from_storage
                    namespace: company.ai

                    tasks:
                      - id: write_instructions
                        type: io.kestra.plugin.core.storage.Write
                        content: |
                          You are a senior code reviewer. When reviewing code:
                          1. Check for security vulnerabilities
                          2. Ensure proper error handling
                          3. Verify naming conventions are followed
                          4. Flag any code duplication

                      - id: agent
                        type: io.kestra.plugin.ai.agent.AIAgent
                        prompt: Review this Python function - "def add(a, b): return a + b"
                        provider:
                          type: io.kestra.plugin.ai.provider.GoogleGemini
                          modelName: gemini-2.5-flash
                          apiKey: "{{ secret('GEMINI_API_KEY') }}"
                        tools:
                          - type: io.kestra.plugin.ai.tool.Skill
                            skills:
                              - name: code_review_expert
                                description: Expert code reviewer with strict guidelines
                                contentUri: "{{ outputs.write_instructions.uri }}"
                """
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
    @PluginProperty(group = "main")
    private List<SkillDefinition> skills;

    @Override
    public Map<ToolSpecification, ToolExecutor> tool(RunContext runContext, Map<String, Object> additionalVariables) throws Exception {
        var skillList = ListUtils.emptyOnNull(skills).stream()
            .map(def -> buildSkill(runContext, additionalVariables, def))
            .toList();

        if (skillList.isEmpty()) {
            throw new IllegalArgumentException("At least one skill must be defined");
        }

        // Validate no duplicate skill names
        var seenNames = new HashSet<String>();
        for (var skill : skillList) {
            if (!seenNames.add(skill.name())) {
                throw new IllegalArgumentException("Duplicate skill name: '" + skill.name() + "'. Each skill must have a unique name.");
            }
        }

        var skills = Skills.from(skillList);
        var invocationContext = InvocationContext.builder().build();
        var userMessage = UserMessage.from("placeholder");
        var result = skills.toolProvider().provideTools(ToolProviderRequest.builder()
            .invocationContext(invocationContext)
            .userMessage(userMessage)
            .build());
        return result.tools();
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
                    resolvedContent = new String(file.readAllBytes(), StandardCharsets.UTF_8);
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

    @Getter
    @Builder
    @Schema(title = "A skill definition")
    public static class SkillDefinition {
        @Schema(title = "Name of the skill")
        @NotNull
        @PluginProperty(group = "main")
        private Property<String> name;

        @Schema(title = "Description of the skill used by the LLM to decide when to activate it")
        @NotNull
        @PluginProperty(group = "main")
        private Property<String> description;

        @Schema(
            title = "Inline content of the skill",
            description = "Mutually exclusive with 'contentUri'. At least one of 'content' or 'contentUri' must be set."
        )
        @PluginProperty(group = "advanced")
        private Property<String> content;

        @Schema(
            title = "URI to the skill content in Kestra internal storage",
            description = "Mutually exclusive with 'content'. At least one of 'content' or 'contentUri' must be set."
        )
        @PluginProperty(internalStorageURI = true, group = "advanced")
        private Property<String> contentUri;

        @Schema(
            title = "Additional resources attached to this skill",
            description = "Resources the agent can read separately using the 'read_skill_resource' tool."
        )
        @PluginProperty(group = "advanced")
        private List<ResourceDefinition> resources;
    }

    @Getter
    @Builder
    @Schema(title = "A skill resource definition")
    public static class ResourceDefinition {
        @Schema(title = "Relative path of the resource within the skill")
        @NotNull
        @PluginProperty(group = "main")
        private Property<String> relativePath;

        @Schema(title = "Content of the resource")
        @NotNull
        @PluginProperty(group = "main")
        private Property<String> content;
    }
}
