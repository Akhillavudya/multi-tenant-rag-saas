"""Embedding provider implementations.

Deliberately empty of imports. Re-exporting `BGELocalEmbedder` here would be
convenient, but this package's modules import torch — so any `import
app.providers.embeddings` anywhere (including Alembic, the CLI, or a test that
only touches chunking) would drag in ~1GB of RSS and seconds of startup.

The factory imports the concrete module lazily instead; see providers/factory.py.
"""
