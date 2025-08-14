package io.kestra.plugin.ai;

import io.micronaut.http.annotation.*;
import io.micronaut.runtime.server.EmbeddedServer;
import jakarta.inject.Inject;

@Controller
public class RemoteA2AAgentController {
    @Inject
    private EmbeddedServer embeddedServer;

    @Post
    public String post(@Body String data) {
        return """
            {
                    "jsonrpc": "2.0",
                    "id": "req_id",
                    "result": {
                        "kind": "task",
                        "id": "001",
                        "contextId": "002",
                        "status": {
                            "state": "completed"
                        },
                        "artifacts": [{
                            "artifactId": "001",
                            "parts": [{
                                "kind": "text",
                                "text": "Hello World"
                            }]
                        }]
                    }
                }
            """;
    }

    @Get("/.well-known/agent-card.json")
    public String agentCard() {
        return """
            {
                    "name": "A2A Demo Agent",
                    "description": "Demo JSON-RPC agent that echoes the last two words of user text.",
                    "url": "%s",
                    "version": "0.1.0",
                    "defaultInputModes": ["text"],
                    "defaultOutputModes": ["text"],
                    "capabilities": {
                        "streaming": false
                    },
                    "skills": [{
                        "id": "hello_world",
                        "name": "Returns hello world",
                        "description": "just returns hello world",
                        "tags": ["hello"]
                    }],
                    "supportsAuthenticatedExtendedCard": false
                }""".formatted(embeddedServer.getURI().toString());
    }
}
