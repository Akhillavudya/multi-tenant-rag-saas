"""LLM provider implementations.

Import-free, for the same reason as the embeddings and rerank packages: each
module here imports a different vendor SDK, and re-exporting them would make
every process pay for all five vendors regardless of which one is configured.

The factory imports the selected module lazily; see providers/factory.py.
"""
