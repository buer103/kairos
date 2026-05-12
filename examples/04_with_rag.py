"""
04 — RAG Integration
====================

Give your agent domain knowledge via Retrieval-Augmented Generation (RAG).
The agent can search your documents before answering questions.

This example:
  1. Creates a vector store from sample documents
  2. Registers a rag_search tool that queries it
  3. Agent uses rag_search when it needs factual context

Requires:
    pip install chromadb          # or: pip install kairos-agent[rag]

Run:
    python examples/04_with_rag.py
"""

import os
import tempfile
from kairos import Agent
from kairos.providers.base import ModelConfig
from kairos.infra.rag.vector_store import VectorStore
from kairos.infra.rag.adapters import TextAdapter
from kairos.tools.rag_search import rag_search, set_rag_store

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-your-key-here"

# ── 1. Build a knowledge base ───────────────────────────────────
# In real code: load your docs, PDFs, or database records

documents = [
    {
        "id": "doc-1",
        "text": "Kairos Agent Framework v0.16.0 supports 20 middleware layers, "
                "23 built-in tools, and 11 platform adapters.",
    },
    {
        "id": "doc-2",
        "text": "The Gateway server runs on port 8080 by default. "
                "You can change it in config.yaml under gateway.port.",
    },
    {
        "id": "doc-3",
        "text": "To add a custom tool, use the @register_tool decorator. "
                "Tools auto-register into the global registry.",
    },
    {
        "id": "doc-4",
        "text": "StatefulAgent supports session persistence with FileSessionBackend "
                "or RedisSessionBackend for production.",
    },
]

# Create vector store and index documents
store = VectorStore(persist_directory=tempfile.mkdtemp(prefix="kairos-rag-"))
adapter = TextAdapter()
for doc in documents:
    chunks = adapter.chunk(doc["text"], chunk_size=200)
    store.add_documents(
        ids=[f"{doc['id']}-{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"source": doc["id"]} for _ in chunks],
    )

# Wire the store into the rag_search tool
set_rag_store(store)

# ── 2. Create an agent that uses RAG ────────────────────────────
agent = Agent(
    model=ModelConfig(api_key=API_KEY),
    agent_name="DocsBot",
    role_description=(
        "You are a documentation assistant. "
        "Use rag_search to look up information about Kairos. "
        "Always cite the source document ID in your answer. "
        "If rag_search returns nothing, say you don't know."
    ),
    rag_store=store,  # also passed here for middleware awareness
)

# ── 3. Ask questions that require document knowledge ────────────
questions = [
    "How many middleware layers does Kairos have?",
    "How do I add a custom tool?",
    "What port does the Gateway run on?",
]

for q in questions:
    print(f"\nQ: {q}")
    result = agent.run(q)
    print(f"A: {result['content']}")
