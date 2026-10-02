from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "infra"


class MetadataCache:
    """
    Small JSON cache used by infra.

    Layout:

        ~/.cache/infra/
            discovery/
            models/

    Discovery cache contains lightweight Hugging Face
    model metadata.

    Model cache contains detailed metadata fetched for
    individual models.
    """

    def __init__(
        self,
        root: Path | None = None,
        ttl_seconds: int = 24 * 60 * 60,
    ) -> None:

        self.root = root or DEFAULT_CACHE_DIR

        self.ttl_seconds = ttl_seconds

        self.discovery_dir = self.root / "discovery"

        self.models_dir = self.root / "models"

        self.discovery_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _is_fresh(
        self,
        path: Path,
    ) -> bool:

        if not path.exists():
            return False

        age = time.time() - path.stat().st_mtime

        return age < self.ttl_seconds

    def get(
        self,
        path: Path,
    ) -> dict[str, Any] | None:

        if not self._is_fresh(path):
            return None

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except (
            OSError,
            json.JSONDecodeError,
        ):
            return None

        if not isinstance(
            data,
            dict,
        ):
            return None

        return data

    def put(
        self,
        path: Path,
        data: dict[str, Any],
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(path.suffix + ".tmp")

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                sort_keys=True,
            )

        temporary.replace(path)

    def discovery_path(
        self,
        key: str,
    ) -> Path:

        safe_key = key.replace("/", "_").replace(" ", "_")

        return self.discovery_dir / f"{safe_key}.json"

    def model_path(
        self,
        model_id: str,
    ) -> Path:

        safe_key = model_id.replace(
            "/",
            "__",
        )

        return self.models_dir / f"{safe_key}.json"

    def clear(self) -> None:
        """
        Remove all cached metadata.
        """

        for directory in (
            self.discovery_dir,
            self.models_dir,
        ):
            for file in directory.glob("*.json"):
                try:
                    file.unlink()

                except OSError:
                    pass
