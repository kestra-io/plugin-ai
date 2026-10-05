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
import dev.langchain4j.skills.ActivateSkillToolConfig;
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
                          modelName: gemini-3.5-flash-lite
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
                          modelName: gemini-3.5-flash-lite
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
        title = "Skill definitions",
        description = "Structured instruction sets the agent can activate on demand. Each skill needs a name, a description, and either inline `content` or a `contentUri` pointing to Kestra internal storage. No default: this property is required.",
        example = "[{name: \"code-review\", description: \"Reviews a diff for bugs\", content: \"Review the diff and list correctness issues.\"}]"
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

        // Include valid skill names in the tool parameter description to guide the model.
        var skillNames = skillList.stream()
            .map(dev.langchain4j.skills.Skill::name)
            .toList();

        var parameterDescription = "The name of the skill to activate. You must use exactly one of these skill names: "
            + String.join(", ", skillNames);

        var skills = Skills.builder()
            .skills(skillList)
            .activateSkillToolConfig(
                ActivateSkillToolConfig.builder()
                    .parameterDescription(parameterDescription)
                    .build()
            )
            .build();

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
        @Schema(
            title = "Skill name",
            description = "Identifier the LLM uses to activate the skill. No default: this property is required.",
            example = "code-review"
        )
        @NotNull
        @PluginProperty(group = "main")
        private Property<String> name;

        @Schema(
            title = "Skill description",
            description = "Natural-language summary of what the skill does, used by the LLM to decide when to activate it. No default: this property is required.",
            example = "Reviews a code diff and reports correctness issues."
        )
        @NotNull
        @PluginProperty(group = "main")
        private Property<String> description;

        @Schema(
            title = "Inline skill content",
            description = "Instructions making up the skill, written inline. Mutually exclusive with `contentUri`; exactly one of the two must be set.",
            example = "Review the provided diff and list any correctness issues you find."
        )
        @PluginProperty(group = "advanced")
        private Property<String> content;

        @Schema(
            title = "Skill content URI",
            description = "Kestra internal storage URI of a file holding the skill instructions. Mutually exclusive with `content`; exactly one of the two must be set.",
            example = "{{ outputs.download_skill.uri }}"
        )
        @PluginProperty(internalStorageURI = true, group = "advanced")
        private Property<String> contentUri;

        @Schema(
            title = "Skill resources",
            description = "Extra files attached to the skill, which the agent reads on demand through the `read_skill_resource` tool rather than receiving them upfront. Not set by default.",
            example = "[{relativePath: \"checklist.md\", content: \"- Check error handling\\n- Check test coverage\"}]"
        )
        @PluginProperty(group = "advanced")
        private List<ResourceDefinition> resources;
    }

    @Getter
    @Builder
    @Schema(title = "A skill resource definition")
    public static class ResourceDefinition {
        @Schema(
            title = "Resource relative path",
            description = "Path identifying the resource within the skill, as the agent refers to it when reading the file. No default: this property is required.",
            example = "checklist.md"
        )
        @NotNull
        @PluginProperty(group = "main")
        private Property<String> relativePath;

        @Schema(
            title = "Resource content",
            description = "Body of the resource file, returned verbatim when the agent reads it. No default: this property is required.",
            example = """
                - Check error handling
                - Check test coverage"""
        )
        @NotNull
        @PluginProperty(group = "main")
        private Property<String> content;
    }
}
