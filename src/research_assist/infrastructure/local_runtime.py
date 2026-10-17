"""Serialize local embedding/reranking forwards even when an asyncio task is cancelled.

Threads cannot be interrupted by cancelling their awaiting coroutine. The worker
must hold the lock until its actual inference ends. This is per-process, not a
GPU memory allocator: loaded model weights remain resident.
"""

from threading import Lock

INFERENCE_LOCK = Lock()
