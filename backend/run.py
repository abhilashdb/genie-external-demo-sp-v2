"""Convenience launcher: `python -m backend.run`.

Starts uvicorn bound to $BACKEND_HOST:$BACKEND_PORT with reload enabled.
"""

from __future__ import annotations

import uvicorn

from .config import settings


def main() -> None:
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
