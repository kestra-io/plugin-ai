@PluginSubGroup(
    description = "Tasks that provision embeddings stores (Chroma, PGVector, Pinecone, Redis, Qdrant, etc.) for chunk storage and similarity search powering RAG flows.",
    categories = { PluginSubGroup.PluginCategory.AI, PluginSubGroup.PluginCategory.DATABASE }
)
package io.kestra.plugin.ai.embeddings;

import io.kestra.core.models.annotations.PluginSubGroup;
