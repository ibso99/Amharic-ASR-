from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import librosa
import datasets
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from Logger import LOGGER
from whisper_utilities import (
    audit_transcripts,
    load_and_validate_manifest,
    normalize_path,
    save_manifest,
    split_manifest_dataframe,
)

logger = LOGGER.getChild("dataset_pipeline")


def create_processor(
        model_name_or_path: str,
        *,
        language: str = "amharic",
        task: str = "transcribe",
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> WhisperProcessor:
    processor = WhisperProcessor.from_pretrained(
        model_name_or_path,
        language=language,
        task=task,
        cache_dir=cache_dir,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    logger.info(
        "Loaded WhisperProcessor | model=%s | language=%s | task=%s | sampling_rate=%s",
        model_name_or_path,
        language,
        task,
        processor.feature_extractor.sampling_rate,
    )

    return processor


def create_model(
    model_name_or_path: str,
    *,
    processor: WhisperProcessor,
    language: str = "amharic",
    task: str = "transcribe",
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    apply_spec_augment: bool = False,
    freeze_feature_encoder: bool = False,
    freeze_encoder: bool = False,
    use_gradient_checkpointing: bool = False,
) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    model.config.apply_spec_augment = apply_spec_augment

    if model.config.decoder_start_token_id is None:
        raise ValueError("model.config.decoder_start_token_id is None")

    # Whisper multilingual prompting.
    try:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language,
            task=task,
        )
    except Exception:
        forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
            language=language,
            task=task,
        )

    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.language = language
    model.generation_config.task = task
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []

    if freeze_feature_encoder:
        # Whisper does not have a Wav2Vec2-style feature encoder.
        # The closest low-level acoustic front-end is conv1/conv2 in the encoder.
        frozen = 0
        for name, param in model.model.encoder.named_parameters():
            if name.startswith("conv1") or name.startswith("conv2"):
                param.requires_grad = False
                frozen += param.numel()
        logger.info("Frozen Whisper encoder convolutional front-end parameters: %d", frozen)

    if freeze_encoder:
        if hasattr(model, "freeze_encoder"):
            model.freeze_encoder()
        else:
            for param in model.model.encoder.parameters():
                param.requires_grad = False
        logger.info("Frozen full Whisper encoder.")

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Enabled gradient checkpointing.")
    else:
        model.config.use_cache = False

    logger.info(
        "Loaded Whisper model | model=%s | spec_augment=%s | freeze_feature_encoder=%s | freeze_encoder=%s",
        model_name_or_path,
        apply_spec_augment,
        freeze_feature_encoder,
        freeze_encoder,
    )

    return model


def dataset_from_dataframe(df: pd.DataFrame) -> Dataset:
    """
    Convert a validated manifest DataFrame into a Hugging Face Dataset.

    We keep audio_path as a plain string and load audio manually during
    preprocessing. This avoids fragile pyarrow casting from string to Audio.
    """
    if "audio_path" not in df.columns:
        raise ValueError("DataFrame must contain 'audio_path' column")

    if "transcript" not in df.columns and "text" not in df.columns:
        raise ValueError("DataFrame must contain either 'transcript' or 'text' column")

    df = df.copy()

    if "text" not in df.columns:
        df["text"] = df["transcript"].astype(str)

    df["audio_path"] = df["audio_path"].astype(str)
    df["text"] = df["text"].astype(str)

    return Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)


def load_local_manifest_datasets(
                *,
                train_csv: Optional[str] = None,
                eval_csv: Optional[str] = None,
                test_csv: Optional[str] = None,
                full_csv: Optional[str] = None,
                split_column: str = "split",
                audio_path_column: str = "audio_path",
                text_column: str = "transcript",
                duration_column: str = "duration_seconds",
                sampling_rate_column: str = "sample_rate",
                target_sampling_rate: int = 16000,
                min_duration_s: float = 0.5,
                max_duration_s: float = 30.0,
                strict_transcript_chars: bool = False,
                do_lower_case: bool = False,
                validate_audio_files: bool = True,
                recompute_durations: bool = False,
                eval_size: float = 0.1,
                test_size: float = 0.0,
                seed: int = 42,
                generated_split_dir: Optional[str] = None,
            ) -> DatasetDict:
    """
    Load local CSV manifests into a DatasetDict with Audio columns.
    Supports either:
    - train_csv + eval_csv [+ test_csv]
    - full_csv with split column or random generated splits
    """
    if full_csv:
        full_df = load_and_validate_manifest(
            full_csv,
            audio_path_column=audio_path_column,
            text_column=text_column,
            split_column=split_column,
            duration_column=duration_column,
            sampling_rate_column=sampling_rate_column,
            target_sampling_rate=target_sampling_rate,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            strict_transcript_chars=strict_transcript_chars,
            do_lower_case=do_lower_case,
            validate_audio_files=validate_audio_files,
            recompute_durations=recompute_durations,
        )

        splits = split_manifest_dataframe(
            full_df,
            split_column=split_column,
            eval_size=eval_size,
            test_size=test_size,
            seed=seed,
        )

        if generated_split_dir:
            out_dir = normalize_path(generated_split_dir, "generated_split_dir")
            out_dir.mkdir(parents=True, exist_ok=True)
            for split_name, split_df in splits.items():
                if len(split_df) > 0:
                    save_manifest(split_df, str(out_dir / f"{split_name}.csv"))

        train_df = splits["train"]
        eval_df = splits["eval"]
        test_df = splits.get("test", pd.DataFrame())

    else:
        if not train_csv or not eval_csv:
            raise ValueError("Provide either full_csv or both train_csv and eval_csv")

        train_df = load_and_validate_manifest(
            train_csv,
            audio_path_column=audio_path_column,
            text_column=text_column,
            split_column=split_column,
            duration_column=duration_column,
            sampling_rate_column=sampling_rate_column,
            target_sampling_rate=target_sampling_rate,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            strict_transcript_chars=strict_transcript_chars,
            do_lower_case=do_lower_case,
            validate_audio_files=validate_audio_files,
            recompute_durations=recompute_durations,
        )

        eval_df = load_and_validate_manifest(
            eval_csv,
            audio_path_column=audio_path_column,
            text_column=text_column,
            split_column=split_column,
            duration_column=duration_column,
            sampling_rate_column=sampling_rate_column,
            target_sampling_rate=target_sampling_rate,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            strict_transcript_chars=strict_transcript_chars,
            do_lower_case=do_lower_case,
            validate_audio_files=validate_audio_files,
            recompute_durations=recompute_durations,
        )

        if test_csv:
            test_df = load_and_validate_manifest(
                test_csv,
                audio_path_column=audio_path_column,
                text_column=text_column,
                split_column=split_column,
                duration_column=duration_column,
                sampling_rate_column=sampling_rate_column,
                target_sampling_rate=target_sampling_rate,
                min_duration_s=min_duration_s,
                max_duration_s=max_duration_s,
                strict_transcript_chars=strict_transcript_chars,
                do_lower_case=do_lower_case,
                validate_audio_files=validate_audio_files,
                recompute_durations=recompute_durations,
            )
        else:
            test_df = pd.DataFrame()

    audit_transcripts(train_df["text"].tolist(), strict_chars=strict_transcript_chars)

    dataset_dict = DatasetDict()
    dataset_dict["train"] = dataset_from_dataframe(train_df)
    dataset_dict["eval"] = dataset_from_dataframe(eval_df)

    if len(test_df) > 0:
        dataset_dict["test"] = dataset_from_dataframe(test_df)

    logger.info(
        "Loaded local datasets | train=%d | eval=%d | test=%d",
        len(dataset_dict["train"]),
        len(dataset_dict["eval"]),
        len(dataset_dict["test"]) if "test" in dataset_dict else 0,
    )

    return dataset_dict


def preprocess_dataset(
                    raw_datasets: DatasetDict,
                     *, 
                    processor: WhisperProcessor, 
                    target_sampling_rate: int = 16000,
                    preprocessing_num_workers: Optional[int] = None,
                    forward_attention_mask: bool = False,
                    max_label_length: Optional[int] = None,
                    max_train_samples: Optional[int] = None,
                    max_eval_samples: Optional[int] = None,
                    max_test_samples: Optional[int] = None,
                ) -> DatasetDict:
    """
    Lightweight preprocessing for production training.

    We do NOT precompute Whisper log-Mel features here.
    Audio is loaded and converted to features inside the collator batch-by-batch.

    This avoids RAM/cache blow-up on large local ASR datasets.
    """

    datasets_to_process = DatasetDict()

    for split_name, dataset in raw_datasets.items():
        if split_name == "train" and max_train_samples:
            dataset = dataset.select(range(min(max_train_samples, len(dataset))))
        elif split_name == "eval" and max_eval_samples:
            dataset = dataset.select(range(min(max_eval_samples, len(dataset))))
        elif split_name == "test" and max_test_samples:
            dataset = dataset.select(range(min(max_test_samples, len(dataset))))

        datasets_to_process[split_name] = dataset

    if max_label_length is not None:
        logger.info(
            "Filtering examples with Whisper label length > %d tokens",
            max_label_length,
        )

        def add_label_length(batch):
            text = str(batch.get("text", "")).strip()

            if not text:
                batch["label_length"] = 0
                batch["valid_label_length"] = False
                return batch

            label_ids = processor.tokenizer(text).input_ids
            batch["label_length"] = len(label_ids)
            batch["valid_label_length"] = len(label_ids) <= max_label_length

            return batch

        before_counts = {
            split_name: len(dataset)
            for split_name, dataset in datasets_to_process.items()
        }

        datasets_to_process = datasets_to_process.map(
            add_label_length,
            num_proc=None,
            desc="Checking Whisper label lengths",
        )

        datasets_to_process = datasets_to_process.filter(
            lambda valid_label_length: bool(valid_label_length),
            input_columns=["valid_label_length"],
            num_proc=None,
            desc="Filtering long Whisper labels",
        )

        after_counts = {
            split_name: len(dataset)
            for split_name, dataset in datasets_to_process.items()
        }

        dropped_counts = {
            split_name: before_counts[split_name] - after_counts[split_name]
            for split_name in before_counts
        }

        logger.info(
            "Label-length filtering complete | before=%s | after=%s | dropped=%s",
            before_counts,
            after_counts,
            dropped_counts,
        )

    logger.info(
        "Using lazy batch-time Whisper feature extraction | datasets=%s",
        {name: len(ds) for name, ds in datasets_to_process.items()},
    )

    return datasets_to_process

def make_sanity_overfit_dataset(raw_datasets: DatasetDict,*,
                            num_examples: int = 2,
                            ) -> DatasetDict:
    if "train" not in raw_datasets:
        raise ValueError("raw_datasets must contain train split for sanity overfit")

    n = min(num_examples, len(raw_datasets["train"]))
    tiny = raw_datasets["train"].select(range(n))

    logger.warning(
        "SANITY OVERFIT ENABLED: using the same %d example(s) for train/eval/test",
        n,
    )

    out = DatasetDict()
    out["train"] = tiny
    out["eval"] = tiny
    out["test"] = tiny

    return out