package io.kestra.plugin.ai.langdock;

/**
 * The Langdock Completion API route used to reach a model. Claude models are only reachable through
 * {@link #ANTHROPIC}; OpenAI/Azure OpenAI-backed models are reached through {@link #OPENAI}.
 */
public enum LangdockModelFamily {
    OPENAI,
    ANTHROPIC
}
