package io.kestra.plugin.ai.provider;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.KeyPairGenerator;
import java.util.Base64;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.Isolated;

import io.kestra.core.context.TestRunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.plugin.ai.domain.ChatConfiguration;

import dev.langchain4j.model.chat.ChatModel;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.assertThat;

// Changes the user.home system property: run alone, not in parallel with other test classes.
@Isolated
@KestraTest
class OciGenAITest {

    @Inject
    private TestRunContextFactory runContextFactory;

    private String previousUserHome;

    // Throwaway ~/.oci/config with a generated key; building the model does not call OCI.
    @BeforeEach
    void setUp(@TempDir Path home) throws Exception {
        var generator = KeyPairGenerator.getInstance("RSA");
        generator.initialize(2048);
        var pem = "-----BEGIN PRIVATE KEY-----\n"
            + Base64.getMimeEncoder(64, "\n".getBytes()).encodeToString(generator.generateKeyPair().getPrivate().getEncoded())
            + "\n-----END PRIVATE KEY-----\n";

        var ociDir = Files.createDirectories(home.resolve(".oci"));
        Files.writeString(ociDir.resolve("key.pem"), pem);
        Files.write(
            ociDir.resolve("config"), List.of(
                "[DEFAULT]",
                "user=ocid1.user.oc1..aaaaaaaatest",
                "fingerprint=aa:bb:cc:dd:ee:ff:00:11:22:33:44:55:66:77:88:99",
                "tenancy=ocid1.tenancy.oc1..aaaaaaaatest",
                "region=us-chicago-1",
                "key_file=" + ociDir.resolve("key.pem")
            )
        );

        previousUserHome = System.getProperty("user.home");
        System.setProperty("user.home", home.toString());
    }

    @AfterEach
    void tearDown() {
        System.setProperty("user.home", previousUserHome);
    }

    @Test
    void chatModel_shouldNotRequireTopK() throws Exception {
        assertThat(chatModel("meta.llama-3.3-70b-instruct")).isNotNull();
    }

    @Test
    void chatModel_givenCohereModel_shouldNotRequireTopK() throws Exception {
        assertThat(chatModel("cohere.command-r-plus")).isNotNull();
    }

    private ChatModel chatModel(String modelName) throws Exception {
        var provider = OciGenAI.builder()
            .type(OciGenAI.class.getName())
            .modelName(Property.ofValue(modelName))
            .compartmentId(Property.ofValue("ocid1.compartment.oc1..aaaaaaaatest"))
            .region(Property.ofValue("us-chicago-1"))
            .build();

        return provider.chatModel(runContextFactory.of(Map.of()), ChatConfiguration.empty());
    }
}
