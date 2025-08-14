package io.kestra.plugin.ai.spi;

import dev.langchain4j.spi.prompt.PromptTemplateFactory;

import java.util.Map;

import static dev.langchain4j.internal.Exceptions.illegalArgument;
import static dev.langchain4j.internal.ValidationUtils.ensureNotBlank;

/**
 * Custom TemplateFactory that creates a lenient template that allows the template to contain a variable that is not passed to the template.
 * This is because we use <code>{{variable}}</code> in our doc and blueprints for Pebble variables, so when a prompt contains that, it
 * didn't always means it's coming from a prompt variable, but can come from some content retrieved (RAG for ex).
 */
public class LenientTemplateFactory implements PromptTemplateFactory {
    @Override
    public Template create(Input input) {
        return new LenientTemplate(input.getTemplate());
    }

    // This is more or less the same as the DefaultTemplate but in a lenient version that didn't check that all variables are set.
    // This is because we use `{{variables}}` a lot in our doc and blueprint as this is how variables are defined in Pebble.
    static class LenientTemplate implements Template {
        private final String template;

        public LenientTemplate(String template) {
            this.template = ensureNotBlank(template, "template");
        }

        @Override
        public String render(Map<String, Object> variables) {
            String result = template;
            for (Map.Entry<String, Object> entry : variables.entrySet()) {
                result = replaceAll(result, entry.getKey(), entry.getValue());
            }

            return result;
        }

        private static String replaceAll(String template, String variable, Object value) {
            if (value == null || value.toString() == null) {
                throw illegalArgument("Value for the variable '%s' is null", variable);
            }
            return template.replace(inDoubleCurlyBrackets(variable), value.toString());
        }

        private static String inDoubleCurlyBrackets(String variable) {
            return "{{" + variable + "}}";
        }
    }
}
