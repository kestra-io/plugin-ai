package io.kestra.plugin.ai.domain;

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
import java.time.Duration;

import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManagerFactory;

import org.bouncycastle.asn1.pkcs.PrivateKeyInfo;
import org.bouncycastle.cert.X509CertificateHolder;
import org.bouncycastle.cert.jcajce.JcaX509CertificateConverter;
import org.bouncycastle.openssl.PEMParser;
import org.bouncycastle.openssl.jcajce.JcaPEMKeyConverter;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.plugins.AdditionalPlugin;
import io.kestra.core.plugins.serdes.PluginDeserializer;
import io.kestra.core.runners.RunContext;

import dev.langchain4j.http.client.jdk.JdkHttpClientBuilder;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.util.Collections;
import java.util.List;
import io.kestra.core.models.annotations.PluginProperty;

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
    @PluginProperty(group = "main")
    private Property<String> modelName;

    @Schema(
        title = "Base URL",
        description = "Custom base URL to override the default endpoint (useful for local tests, WireMock, or enterprise gateways)."
    )
    @PluginProperty(group = "connection")
    protected Property<String> baseUrl;

    @Schema(
        title = "Client PEM certificate content",
        description = "PEM client certificate as text, used to authenticate the connection to enterprise AI endpoints."
    )
    @PluginProperty(group = "advanced")
    private Property<String> clientPem;

    @Schema(
        title = "CA PEM certificate content",
        description = "CA certificate as text, used to verify SSL/TLS connections when using custom endpoints."
    )
    @PluginProperty(group = "advanced")
    private Property<String> caPem;

    @FunctionalInterface
    protected interface ClassLoaderCallable<T> {
        T call() throws IllegalVariableEvaluationException;
    }

    protected <T> T withPluginClassLoader(ClassLoaderCallable<T> callable) throws IllegalVariableEvaluationException {
        ClassLoader previous = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(getClass().getClassLoader());
        try {
            return callable.call();
        } finally {
            Thread.currentThread().setContextClassLoader(previous);
        }
    }

    public final ChatModel chatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        return withPluginClassLoader(() -> buildChatModel(runContext, configuration));
    }

    public final ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout) throws IllegalVariableEvaluationException {
        return withPluginClassLoader(() -> buildChatModel(runContext, configuration, timeout));
    }

    public final ChatModel chatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout, List<ChatModelListener> additionalListeners) throws IllegalVariableEvaluationException {
        return withPluginClassLoader(() -> buildChatModel(runContext, configuration, timeout, additionalListeners));
    }

    public final ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return withPluginClassLoader(() -> buildImageModel(runContext));
    }

    public final EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException {
        return withPluginClassLoader(() -> buildEmbeddingModel(runContext));
    }

    protected ChatModel buildChatModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException {
        return buildChatModel(runContext, configuration, Duration.ofSeconds(120));
    }

    protected ChatModel buildChatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout) throws IllegalVariableEvaluationException {
        return buildChatModel(runContext, configuration, timeout, Collections.emptyList());
    }

    protected abstract ChatModel buildChatModel(RunContext runContext, ChatConfiguration configuration, Duration timeout, List<ChatModelListener> additionalListeners) throws IllegalVariableEvaluationException;

    protected abstract ImageModel buildImageModel(RunContext runContext) throws IllegalVariableEvaluationException;

    protected abstract EmbeddingModel buildEmbeddingModel(RunContext runContext) throws IllegalVariableEvaluationException;

    protected JdkHttpClientBuilder buildHttpClientWithPemIfAvailable(RunContext runContext) throws IllegalVariableEvaluationException {

        String clientPem = runContext.render(this.clientPem).as(String.class).orElse(null);
        String caPem = runContext.render(this.caPem).as(String.class).orElse(null);

        if (clientPem == null && caPem == null) {
            runContext.logger().debug("No PEM certificates provided, using default HttpClient");
            return null;
        }

        try (
            ByteArrayInputStream clientPemStream = clientPem != null ? new ByteArrayInputStream(clientPem.getBytes(StandardCharsets.UTF_8)) : null;
            ByteArrayInputStream caPemStream = caPem != null ? new ByteArrayInputStream(caPem.getBytes(StandardCharsets.UTF_8)) : null
        ) {

            HttpClient.Builder httpClient = withPemCertificate(clientPemStream, caPemStream);

            return new JdkHttpClientBuilder().httpClientBuilder(httpClient);

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

        Certificate[] privateKeyCertificatesChain = new Certificate[] { clientCertificate };

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
