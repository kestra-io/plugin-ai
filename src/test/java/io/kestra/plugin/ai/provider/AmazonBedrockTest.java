package io.kestra.plugin.ai.provider;

import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.domain.ChatConfiguration;

import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@ResourceLock("kestra-h2-flyway")
@KestraTest
class AmazonBedrockTest {
    @Inject
    private TestRunContextFactory runContextFactory;

    private AmazonBedrock provider() {
        return AmazonBedrock.builder()
            .type(AmazonBedrock.class.getName())
            .accessKeyId(Property.ofValue("test-key"))
            .secretAccessKey(Property.ofValue("test-secret"))
            .modelName(Property.ofValue("us.anthropic.claude-3-7-sonnet-20250219-v1:0"))
            .build();
    }

    @Test
    void chatModel_shouldRejectThinkingBudgetEqualToMaxToken() {
        var configuration = ChatConfiguration.builder()
            .thinkingBudgetTokens(Property.ofValue(2048))
            .maxToken(Property.ofValue(2048))
            .build();

        assertThatThrownBy(() -> provider().chatModel(runContextFactory.of(Map.of()), configuration))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("`maxToken` must be greater than `thinkingBudgetTokens`");
    }

    @Test
    void chatModel_shouldAcceptThinkingBudgetBelowMaxToken() throws Exception {
        var configuration = ChatConfiguration.builder()
            .thinkingBudgetTokens(Property.ofValue(1024))
            .maxToken(Property.ofValue(2048))
            .build();

        // the AWS client resolves a region at build time; no network call is made
        var previousRegion = System.getProperty("aws.region");
        System.setProperty("aws.region", "us-east-1");
        try {
            assertThat(provider().chatModel(runContextFactory.of(Map.of()), configuration)).isNotNull();
        } finally {
            if (previousRegion == null) {
                System.clearProperty("aws.region");
            } else {
                System.setProperty("aws.region", previousRegion);
            }
        }
    }
}
