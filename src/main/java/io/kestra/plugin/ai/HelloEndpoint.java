package io.kestra.plugin.ai;

import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.plugins.endpoint.PluginEndpoint;
import io.kestra.core.plugins.endpoint.PluginEndpointRequest;
import io.kestra.core.plugins.endpoint.PluginEndpointResponse;

import java.util.Map;

@Plugin
public class HelloEndpoint implements PluginEndpoint {
    @Override
    public String name() {
        return "hello";
    }

    @Override
    public PluginEndpointResponse handle(PluginEndpointRequest request) {
        String name = request.param("name");
        return PluginEndpointResponse.of(Map.of("message", "hello: " + name));
    }
}
