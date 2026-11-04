package io.kestra.plugin.ai.provider;

import java.util.Map;

import org.junit.jupiter.api.Test;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.domain.ChatConfiguration;

import dev.langchain4j.model.chat.ChatModel;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class WatsonxAITest {

    @Inject
    private TestRunContextFactory runContextFactory;

    @Test
    void chatModel_shouldNotCapOutputTokensByDefault() throws Exception {
        var model = chatModel(ChatConfiguration.empty());

        assertThat(model.defaultRequestParameters().maxOutputTokens()).isNull();
    }

    @Test
    void chatModel_shouldHonourMaxTokenAndTopP() throws Exception {
        var configuration = ChatConfiguration.builder()
            .maxToken(Property.ofValue(2048))
            .topP(Property.ofValue(0.5))
            .build();

        var parameters = chatModel(configuration).defaultRequestParameters();

        assertThat(parameters.maxOutputTokens()).isEqualTo(2048);
        assertThat(parameters.topP()).isEqualTo(0.5);
    }

    // Building the model does not call watsonx.ai.
    private ChatModel chatModel(ChatConfiguration configuration) throws Exception {
        var provider = WatsonxAI.builder()
            .type(WatsonxAI.class.getName())
            .modelName(Property.ofValue("ibm/granite-3-3-8b-instruct"))
            .baseUrl(Property.ofValue("https://us-south.ml.cloud.ibm.com"))
            .apiKey(Property.ofValue("test-key"))
            .projectId(Property.ofValue("test-project"))
            .build();

        return provider.chatModel(runContextFactory.of(Map.of()), configuration);
    }
}
