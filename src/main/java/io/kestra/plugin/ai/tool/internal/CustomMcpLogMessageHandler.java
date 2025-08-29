package io.kestra.plugin.ai.tool.internal;

import dev.langchain4j.mcp.client.logging.McpLogMessage;
import dev.langchain4j.mcp.client.logging.McpLogMessageHandler;
import org.slf4j.Logger;

public class CustomMcpLogMessageHandler implements McpLogMessageHandler {

    private final Logger logger;

    public CustomMcpLogMessageHandler(Logger logger) {
        this.logger = logger;
    }

    @Override
    public void handleLogMessage(McpLogMessage message) {
        if (message.level() == null) {
            logger.warn("Received MCP log message with unknown level: {}", message.data());
            return;
        }
        switch (message.level()) {
            case DEBUG -> logger.debug("MCP logger: {}: {}", message.logger(), message.data());
            case INFO, NOTICE -> logger.info("MCP logger: {}: {}", message.logger(), message.data());
            case WARNING -> logger.warn("MCP logger: {}: {}", message.logger(), message.data());
            case ERROR, CRITICAL, ALERT, EMERGENCY -> logger.error("MCP logger: {}: {}", message.logger(), message.data());
        }
    }
}
