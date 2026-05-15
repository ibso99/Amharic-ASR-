from __future__ import annotations

import logging
from typing import Union

LOGGER = logging.getLogger("amharic_whisper_asr")
LOGGER.addHandler(logging.NullHandler())


def _normalize_log_level(log_level: Union[str, int]) -> int:
    if isinstance(log_level, int):
        return log_level

    if not isinstance(log_level, str):
        raise TypeError("log_level must be a string or integer")

    level_name = log_level.strip().upper()
    if not hasattr(logging, level_name):
        raise ValueError(
            f"Invalid log level: {log_level!r}. "
            "Expected one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )

    level_value = getattr(logging, level_name)
    if not isinstance(level_value, int):
        raise ValueError(f"Invalid log level: {log_level!r}")

    return level_value


def setup_logging(log_level: Union[str, int] = "INFO") -> None:
    level = _normalize_log_level(log_level)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    LOGGER.setLevel(level)