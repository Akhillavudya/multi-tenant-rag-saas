"""Rerank provider implementations.

Kept import-free for the same reason as app/providers/embeddings/__init__.py:
`bge_local` here pulls torch, and anything written in this file runs on every
import beneath the package. `noop` has no heavy dependencies, but re-exporting
only that would be an inconsistent rule that the next person would break.

The factory imports the selected module lazily; see providers/factory.py.
"""
