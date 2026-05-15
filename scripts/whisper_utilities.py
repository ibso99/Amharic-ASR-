from __future__ import annotations

import os
import random
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import librosa
import numpy as np
import pandas as pd
import torch

from Logger import LOGGER

logger = LOGGER.getChild("utilities")


# ============================================================
# Constants
# ============================================================

REQUIRED_MANIFEST_COLUMNS = {"audio_path", "transcript"}

ETHIOPIC_BLOCKS = (
    (0x1200, 0x137F),  # Ethiopic
    (0x1380, 0x139F),  # Ethiopic Supplement
    (0x2D80, 0x2DDF),  # Ethiopic Extended
    (0xAB00, 0xAB2F),  # Ethiopic Extended-A
)

ALLOWED_ASCII_CHARS = set(
    "0123456789 .,!?;:'\"()[]{}-_/&%+*=#@"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int, deterministic: bool = False, warn_only: bool = True) -> None:
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic

    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    except TypeError:
        if deterministic:
            torch.use_deterministic_algorithms(True)


# ============================================================
# Path helpers
# ============================================================

def normalize_path(path_str: str, kind: str) -> Path:
    if not isinstance(path_str, str) or not path_str.strip():
        raise ValueError(f"{kind} must be a non-empty string")

    return Path(path_str).expanduser().resolve(strict=False)


# ============================================================
# Character checks
# ============================================================

def is_ethiopic_char(ch: str) -> bool:
    code = ord(ch)
    return any(start <= code <= end for start, end in ETHIOPIC_BLOCKS)


def is_allowed_transcript_char(ch: str) -> bool:
    if ch == " ":
        return True
    if is_ethiopic_char(ch):
        return True
    if ch in ALLOWED_ASCII_CHARS:
        return True
    return False


# ============================================================
# Text normalization
# ============================================================

def strip_invisible_and_control(text: str) -> str:
    """
    Remove invisible/control characters that often appear in copied transcripts.
    """
    text = unicodedata.normalize("NFC", str(text))

    nuisance_chars = (
        "\u00A0",  # non-breaking space
        "\u200B",  # zero-width space
        "\u200C",  # zero-width non-joiner
        "\u200D",  # zero-width joiner
        "\u2060",  # word joiner
        "\uFEFF",  # BOM
        "\t",
        "\r",
        "\n",
    )

    for bad in nuisance_chars:
        text = text.replace(bad, " ")

    cleaned_chars: List[str] = []

    for ch in text:
        category = unicodedata.category(ch)

        # Cc = control, Cf = format, Cs = surrogate
        if category in {"Cc", "Cf", "Cs"}:
            continue

        cleaned_chars.append(ch)

    return "".join(cleaned_chars)


def normalize_amharic_for_asr(text: str) -> str:
    """
    Normalize Amharic text for fair ASR metric computation.

    This function should be applied to BOTH references and predictions before
    computing WER/CER. It reduces metric inflation caused by inconsistent
    punctuation, spacing, and common Ethiopic orthographic variants.
    """
    text = strip_invisible_and_control(str(text))

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Common Amharic/Ethiopic orthographic variants.
    # These mappings are useful for ASR evaluation, but should be reviewed
    # against your preferred writing standard.
    replacements = {
        # ሀ / ሐ / ኀ series
        "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
        "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",

        # አ / ዐ series
        "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "አ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",

        # ፀ / ጸ series
        "ጸ": "ፀ", "ጹ": "ፁ", "ጺ": "ፂ", "ጻ": "ፃ", "ጼ": "ፄ", "ጽ": "ፅ", "ጾ": "ፆ",
    }

    for src, tgt in replacements.items():
        text = text.replace(src, tgt)

    # Remove punctuation if references are inconsistent.
    text = re.sub(r"[።፣፤፥፦፧፨.,!?;:\"'()\[\]{}]", "", text)

    # Normalize whitespace again
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_training_transcript(text: str) -> str:
    """
    Normalize transcripts used as Whisper training targets.

    This function is stronger than metric-only normalization because it changes
    the target text the model learns to generate. It is designed to reduce the
    fragmented spacing pattern seen in the current Amharic dataset.
    """
    text = normalize_amharic_for_asr(text)

    # Conservative cleanup of fragmented prefix/preposition spacing.
    # Examples:
    #   "የ ደብሩ" -> "የደብሩ"
    #   "በ መምጣቱ" -> "በመምጣቱ"
    #   "ለ መረዳት" -> "ለመረዳት"
    text = re.sub(r"\bየ\s+", "የ", text)
    text = re.sub(r"\bበ\s+", "በ", text)
    text = re.sub(r"\bለ\s+", "ለ", text)
    text = re.sub(r"\bከ\s+", "ከ", text)
    text = re.sub(r"\bእንደ\s+", "እንደ", text)

    # Common one-letter verb-prefix fragmentation.
    # Examples:
    #   "ይ ፈጥራል" -> "ይፈጥራል"
    #   "ሲ ሆኑ" -> "ሲሆኑ"
    #   "ተ ጠቅሷል" -> "ተጠቅሷል"
    text = re.sub(r"\b(ይ|ት|እ|አ|ተ|ነ|ሲ|ሳይ|ሚ|ምን)\s+", r"\1", text)

    # Common suffix/particle cleanup.
    # Examples:
    #   "ዜጎች ም" -> "ዜጎችም"
    #   "ያለ ቁም" -> "ያለቁም"
    text = re.sub(r"\s+(ም|ን|ና|ነት)\b", r"\1", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_amharic_transcript(
    value: Any,
    *,
    strict_chars: bool = False,
    do_lower_case: bool = False,
    apply_training_normalization: bool = True,
) -> str:
    """
    Normalize transcript text for Whisper fine-tuning.

    Steps:
    1. Remove invisible/control characters.
    2. Normalize whitespace.
    3. Optionally remove disallowed characters.
    4. Optionally apply Amharic ASR training normalization.

    Notes:
    - Whisper uses a multilingual tokenizer, so no custom vocabulary is built.
    - `apply_training_normalization=True` standardizes the transcript style used
      as the model target.
    """
    if value is None:
        raise ValueError("Transcript is None")

    if pd.isna(value):
        raise ValueError("Transcript is NaN")

    if not isinstance(value, str):
        value = str(value)

    original = value
    text = strip_invisible_and_control(value)

    if do_lower_case:
        text = text.lower()

    text = re.sub(r"\s+", " ", text).strip()

    if strict_chars:
        removed_chars = sorted(set(ch for ch in text if not is_allowed_transcript_char(ch)))
        text = "".join(ch if is_allowed_transcript_char(ch) else "" for ch in text)
        text = re.sub(r"\s+", " ", text).strip()

        if removed_chars:
            logger.debug(
                "Removed disallowed chars from transcript: %s | original=%r",
                removed_chars[:20],
                original[:200],
            )

    if apply_training_normalization:
        text = normalize_training_transcript(text)

    if not text:
        raise ValueError(
            f"Transcript became empty after normalization: {original[:200]!r}"
        )

    return text


def audit_transcripts(
    transcripts: Sequence[str],
    *,
    strict_chars: bool = False,
    max_examples: int = 10,
) -> None:
    """
    Audit transcript characters and report suspicious/disallowed characters.
    """
    bad_counter: Counter[str] = Counter()
    bad_examples: List[Dict[str, Any]] = []

    for idx, raw_text in enumerate(transcripts):
        text = "" if raw_text is None else str(raw_text)
        text = strip_invisible_and_control(text)
        text = re.sub(r"\s+", " ", text).strip()

        if strict_chars:
            bad_chars = [ch for ch in text if not is_allowed_transcript_char(ch)]
        else:
            bad_chars = []

        if bad_chars:
            bad_counter.update(bad_chars)

            if len(bad_examples) < max_examples:
                bad_examples.append(
                    {
                        "index": idx,
                        "bad_chars": sorted(set(bad_chars)),
                        "text": text[:300],
                    }
                )

    if bad_counter:
        logger.warning(
            "Transcript audit found disallowed characters. Top offenders: %s",
            bad_counter.most_common(20),
        )

        for item in bad_examples:
            logger.warning(
                "BAD TRANSCRIPT EXAMPLE | idx=%d | bad_chars=%s | text=%r",
                item["index"],
                item["bad_chars"],
                item["text"],
            )
    else:
        logger.info("Transcript audit completed.")


# ============================================================
# Audio helpers
# ============================================================

def get_audio_duration_seconds(audio_path: str, sampling_rate: int = 16000) -> float:
    path = normalize_path(audio_path, "audio_path")

    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Audio path is not a file: {path}")

    try:
        duration = librosa.get_duration(path=str(path))
    except Exception:
        audio, _ = librosa.load(str(path), sr=sampling_rate, mono=True)
        duration = len(audio) / sampling_rate

    return float(duration)


# ============================================================
# Manifest loading and validation
# ============================================================

def load_and_validate_manifest(
    csv_path: str,
    *,
    audio_path_column: str = "audio_path",
    text_column: str = "transcript",
    split_column: str = "split",
    duration_column: str = "duration_seconds",
    sampling_rate_column: str = "sample_rate",
    target_sampling_rate: int = 16000,
    min_duration_s: float = 0.5,
    max_duration_s: float = 30.0,
    strict_transcript_chars: bool = False,
    do_lower_case: bool = False,
    validate_audio_files: bool = True,
    recompute_durations: bool = False,
    dropped_report_path: Optional[str] = None,
    apply_training_normalization: bool = True,
) -> pd.DataFrame:
    """
    Load, validate, and clean a local CSV manifest.

    Expected minimum columns:
        audio_path
        transcript

    The returned DataFrame always includes:
        audio_path
        text
        audio
        duration_seconds
        sample_rate
    """
    manifest_path = normalize_path(csv_path, "csv_path")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {manifest_path}")

    if not manifest_path.is_file():
        raise ValueError(f"Manifest path is not a file: {manifest_path}")

    df = pd.read_csv(manifest_path)

    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Manifest {manifest_path} is missing required columns: {sorted(missing_columns)}"
        )

    if audio_path_column not in df.columns:
        raise ValueError(f"audio_path_column {audio_path_column!r} not found in CSV")

    if text_column not in df.columns:
        raise ValueError(f"text_column {text_column!r} not found in CSV")

    logger.info("Loaded manifest %s with shape=%s", manifest_path, df.shape)

    valid_rows: List[Dict[str, Any]] = []
    dropped_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        try:
            audio_path = str(row[audio_path_column]).strip()

            if not audio_path:
                raise ValueError("audio_path is empty")

            path = normalize_path(audio_path, "audio_path")

            if validate_audio_files and not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")

            transcript = normalize_amharic_transcript(
                row[text_column],
                strict_chars=strict_transcript_chars,
                do_lower_case=do_lower_case,
                apply_training_normalization=apply_training_normalization,
            )

            if (
                recompute_durations
                or duration_column not in df.columns
                or pd.isna(row.get(duration_column))
            ):
                duration = get_audio_duration_seconds(
                    str(path),
                    sampling_rate=target_sampling_rate,
                )
            else:
                duration = float(row[duration_column])

            if duration < min_duration_s:
                raise ValueError(
                    f"Audio too short: {duration:.3f}s < {min_duration_s}s"
                )

            if duration > max_duration_s:
                raise ValueError(
                    f"Audio too long: {duration:.3f}s > {max_duration_s}s"
                )

            sample_rate = (
                int(row[sampling_rate_column])
                if sampling_rate_column in df.columns and not pd.isna(row.get(sampling_rate_column))
                else target_sampling_rate
            )

            cleaned = dict(row_dict)

            cleaned[audio_path_column] = str(path)
            cleaned[text_column] = transcript
            cleaned[duration_column] = duration
            cleaned[sampling_rate_column] = sample_rate

            # Standard fields consumed by the dataset pipeline/collator.
            cleaned["audio_path"] = str(path)
            cleaned["audio"] = str(path)
            cleaned["text"] = transcript

            valid_rows.append(cleaned)

        except Exception as exc:
            row_dict["drop_reason"] = str(exc)
            dropped_rows.append(row_dict)

    valid_df = pd.DataFrame(valid_rows)

    if valid_df.empty:
        raise ValueError(
            f"No valid rows found after validating manifest: {manifest_path}"
        )

    if dropped_rows:
        dropped_df = pd.DataFrame(dropped_rows)

        if dropped_report_path:
            output_path = normalize_path(dropped_report_path, "dropped_report_path")
        else:
            output_path = manifest_path.with_suffix(".dropped_rows.csv")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        dropped_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        logger.warning(
            "Dropped %d/%d rows from %s. Report saved to %s",
            len(dropped_rows),
            len(df),
            manifest_path,
            output_path,
        )

    logger.info(
        "Validated manifest %s | kept=%d | dropped=%d",
        manifest_path,
        len(valid_df),
        len(dropped_rows),
    )

    return valid_df


def save_manifest(df: pd.DataFrame, output_path: str) -> None:
    path = normalize_path(output_path, "output_path")
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        tmp_path = Path(tmp.name)
        df.to_csv(tmp, index=False)

    tmp_path.replace(path)

    logger.info("Saved manifest with %d rows to %s", len(df), path)


def split_manifest_dataframe(
    df: pd.DataFrame,
    *,
    split_column: str = "split",
    eval_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Return train/eval/test splits.

    If `split_column` exists and contains train/validation/test labels, it is
    used. Otherwise, random train/eval/test splits are created.
    """
    if split_column in df.columns:
        split_values = df[split_column].astype(str).str.lower().str.strip()

        train_df = df[split_values == "train"].copy()
        eval_df = df[split_values.isin(["validation", "valid", "eval", "dev"])].copy()
        test_df = df[split_values == "test"].copy()

        if len(train_df) > 0 and len(eval_df) > 0:
            if len(test_df) == 0:
                logger.warning("No test split found. Proceeding with train/eval only.")

            return {
                "train": train_df,
                "eval": eval_df,
                "test": test_df,
            }

    if not (0.0 < eval_size < 1.0):
        raise ValueError("eval_size must be between 0 and 1")

    if not (0.0 <= test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")

    if eval_size + test_size >= 1.0:
        raise ValueError("eval_size + test_size must be < 1")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_total = len(shuffled)
    n_test = int(n_total * test_size)
    n_eval = int(n_total * eval_size)

    test_df = (
        shuffled.iloc[:n_test].copy()
        if n_test > 0
        else pd.DataFrame(columns=shuffled.columns)
    )

    eval_df = shuffled.iloc[n_test:n_test + n_eval].copy()
    train_df = shuffled.iloc[n_test + n_eval:].copy()

    return {
        "train": train_df,
        "eval": eval_df,
        "test": test_df,
    }