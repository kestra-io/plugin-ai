package io.kestra.plugin.ai.provider.github;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

/**
 * Unit tests for GitHubProvider that mock GitHubModelsClient so tests don't hit network/Docker.
 */
public class GitHubProviderTest {
    @Test
    public void chatParsesMessageContent() throws Exception {
        // Arrange
        GitHubModelsConfig cfg = GitHubModelsConfig.builder()
            .token("dummy")
            .defaultModel("gpt-4.1-mini")
            .build();

        // Create provider instance
        GitHubProvider provider = new GitHubProvider(cfg);

        // Mock client and response
        GitHubModelsClient mockClient = mock(GitHubModelsClient.class);
        Map<String, Object> choice = Map.of(
            "message", Map.of("content", "hello from mock")
        );
        when(mockClient.chat(anyString(), anyList(), any(), anyMap()))
            .thenReturn("hello from mock");

        // Inject mock client into provider (reflection because client is private)
        Field clientField = GitHubProvider.class.getDeclaredField("client");
        clientField.setAccessible(true);
        clientField.set(provider, mockClient);

        // Act
        String out = provider.chat("gpt-4.1-mini", "some prompt");

        // Assert
        assertEquals("hello from mock", out);
        verify(mockClient, times(1)).chat(eq("gpt-4.1-mini"), anyList(), eq(null), eq(null));
    }

    @Test
    public void embeddingsReturnsList() throws Exception {
        // Arrange
        GitHubModelsConfig cfg = GitHubModelsConfig.builder()
            .token("dummy")
            .defaultModel("embed-model")
            .build();
        GitHubProvider provider = new GitHubProvider(cfg);

        GitHubModelsClient mockClient = mock(GitHubModelsClient.class);
        when(mockClient.embeddings(anyString(), anyString()))
            .thenReturn(List.of(0.1, 0.2));

        Field clientField = GitHubProvider.class.getDeclaredField("client");
        clientField.setAccessible(true);
        clientField.set(provider, mockClient);

        // Act
        var resp = provider.embeddings("embed-model", "text to embed");

        // Assert
        assertEquals(2, resp.size());
        verify(mockClient, times(1)).embeddings(eq("embed-model"), eq("text to embed"));
    }
}
