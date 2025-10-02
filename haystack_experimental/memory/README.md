# Memory Components

This directory contains Haystack components for implementing agent memory capabilities.

## Components

### MemoryStore Protocol (`protocol.py`)
Defines the interface for pluggable memory storage backends.

### Mem0MemoryStore (`mem0_store.py`)
Implementation using Mem0 as the backend storage service.

### MemoryRetriever (`memory_retriever.py`)
Component for retrieving relevant memories based on queries.

### MemoryWriter (`memory_writer.py`)
Component for storing chat messages as memories.

## Usage

### Examples

1. **Basic Memory Pipeline** (`examples/memory_pipeline_example.py`)
   - Simple memory storage and retrieval
   - Different memory types demonstration

2. **Agent Memory Integration** (`examples/agent_memory_pipeline_example.py`)
   - Complete agent with memory capabilities
   - Memory-aware conversations
   - Preference learning and recall
   - Session persistence

3. **Simple Agent Memory** (`examples/simple_agent_memory_example.py`)
   - Minimal agent memory integration
   - Direct pipeline structure
   - Easy to understand and modify

## Memory Types

Memories are stored as ChatMessage objects with metadata:
- `memory_type`: "semantic" (facts/preferences) or "episodic" (experiences)
- `user_id`: User identifier for scoping
- `memory_id`: Unique identifier (set by storage backend)

## Requirements

- `pip install mem0ai` for Mem0MemoryStore
- MEM0_API_KEY environment variable
