package io.kestra.plugin.ai.domain;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import dev.langchain4j.http.client.HttpClientBuilderLoader;
import dev.langchain4j.http.client.jdk.JdkHttpClientBuilder;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.plugins.AdditionalPlugin;
import io.kestra.core.plugins.serdes.PluginDeserializer;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import org.bouncycastle.asn1.pkcs.PrivateKeyInfo;
import org.bouncycastle.cert.X509CertificateHolder;
import org.bouncycastle.cert.jcajce.JcaX509CertificateConverter;
import org.bouncycastle.openssl.PEMParser;
import org.bouncycastle.openssl.jcajce.JcaPEMKeyConverter;

import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManagerFactory;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.security.*;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;

@Plugin
@SuperBuilder(toBuilder = true)
@Getter
@NoArgsConstructor
// IMPORTANT: The abstract plugin base class must define using the PluginDeserializer,
// AND concrete subclasses must be annotated by @JsonDeserialize() to avoid StackOverflow.
@JsonDeserialize(using = PluginDeserializer.class)
public abstract class ModelProvider extends AdditionalPlugin {

    @Schema(title = "Model name")
    @NotNull
    private Property<String> modelName;

    @Schema(
        title = "Base URL",
        description = "Custom base URL to override the default endpoint (useful for local tests, WireMock, or enterprise gateways)."
    )
    protected Property<String> baseUrl;

    @Schema(
        title = "Client PEM certificate content",
        description = "PEM client certificate as text, used to authenticate the connection to enterprise AI endpoints."
    )
    private Property<String> clientPem;

    @Schema(
        title = "CA PEM certificate content",
        description = "CA certificate as text, used to verify SSL/TLS connections when using custom endpoints."
    )
    private Property<String> caPem;

    public abstract ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException;

    public abstract ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException;

    public abstract EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException;

    protected JdkHttpClientBuilder buildHttpClientWithPemIfAvailable(RunContext runContext) throws IllegalVariableEvaluationException {

        String clientPem = runContext.render(this.clientPem).as(String.class).orElse(null);
        String caPem = runContext.render(this.caPem).as(String.class).orElse(null);

        if (clientPem == null && caPem == null) {
            runContext.logger().debug("No PEM certificates provided, using default HttpClient");
            return null;
        }

        try (ByteArrayInputStream clientPemStream =
                 clientPem != null ? new ByteArrayInputStream(clientPem.getBytes(StandardCharsets.UTF_8)) : null;
             ByteArrayInputStream caPemStream =
                 caPem != null ? new ByteArrayInputStream(caPem.getBytes(StandardCharsets.UTF_8)) : null) {

            HttpClient.Builder httpClient = withPemCertificate(clientPemStream, caPemStream);

            return ((JdkHttpClientBuilder) HttpClientBuilderLoader.loadHttpClientBuilder())
                .httpClientBuilder(httpClient);

        } catch (Exception e) {
            runContext.logger().error("Error while setting up mTLS HTTP client: {}", e.getMessage(), e);
            throw new IllegalArgumentException("Exception while trying to setup AI Service certificates", e);
        }
    }

    public static HttpClient.Builder withPemCertificate(InputStream clientPemIs, InputStream caPem)
        throws KeyStoreException, IOException, NoSuchAlgorithmException, CertificateException, KeyManagementException, UnrecoverableKeyException {

        PrivateKey privateKey = null;
        Certificate clientCertificate = null;

        // Parse the PEM content to extract certificate and private key
        try (PEMParser pemParser = new PEMParser(new InputStreamReader(clientPemIs))) {
            JcaPEMKeyConverter keyConverter = new JcaPEMKeyConverter();
            JcaX509CertificateConverter certConverter = new JcaX509CertificateConverter();
            Object object;
            while ((object = pemParser.readObject()) != null) {
                if (object instanceof PrivateKeyInfo privateKeyInfo) {
                    privateKey = keyConverter.getPrivateKey(privateKeyInfo);
                } else if (object instanceof X509CertificateHolder) {
                    clientCertificate = certConverter.getCertificate((X509CertificateHolder) object);
                }
            }
        }

        KeyStore keyStore = KeyStore.getInstance("PKCS12");
        keyStore.load(null, null);

        Certificate[] privateKeyCertificatesChain = new Certificate[]{clientCertificate};

        if (caPem != null) {
            CertificateFactory cf = CertificateFactory.getInstance("X.509");
            keyStore.setCertificateEntry("ca", cf.generateCertificate(caPem));
        }

        keyStore.setKeyEntry("client-key", privateKey, "".toCharArray(), privateKeyCertificatesChain);

        KeyManagerFactory kmf = KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm());
        kmf.init(keyStore, "".toCharArray());

        TrustManagerFactory tmf = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
        tmf.init(keyStore);

        SSLContext sslContext = SSLContext.getInstance("TLS");
        sslContext.init(kmf.getKeyManagers(), tmf.getTrustManagers(), new SecureRandom());

        return HttpClient.newBuilder().sslContext(sslContext);
    }
}
