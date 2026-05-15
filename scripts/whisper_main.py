from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional
import json
import sys

import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

from whisper_dataset_pipeline import (
    create_model,
    create_processor,
    load_local_manifest_datasets,
    make_sanity_overfit_dataset,
    preprocess_dataset,
)
from Logger import LOGGER, setup_logging
from whisper_train import create_trainer, save_training_config, train_and_evaluate
from whisper_utilities import set_seed
import torch.distributed as dist

logger = LOGGER.getChild("main")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"Invalid boolean value for {name}: {value!r}")


def env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def env_path(name: str, default: Optional[str] = None) -> Optional[str]:
    value = env_str(name, default)
    if value is None:
        return None
    return str(Path(value).expanduser().resolve(strict=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production-grade Whisper-small fine-tuning for Amharic ASR"
    )

    # Data
    parser.add_argument("--train-csv", type=str, default=env_path("TRAIN_CSV"))
    parser.add_argument("--eval-csv", type=str, default=env_path("EVAL_CSV"))
    parser.add_argument("--test-csv", type=str, default=env_path("TEST_CSV"))
    parser.add_argument("--full-csv", type=str, default=env_path("FULL_CSV"))

    parser.add_argument("--audio-path-column", type=str, default=env_str("AUDIO_PATH_COLUMN", "audio_path"))
    parser.add_argument("--text-column", type=str, default=env_str("TEXT_COLUMN", "transcript"))
    parser.add_argument("--split-column", type=str, default=env_str("SPLIT_COLUMN", "split"))
    parser.add_argument("--duration-column", type=str, default=env_str("DURATION_COLUMN", "duration_seconds"))
    parser.add_argument("--sampling-rate-column", type=str, default=env_str("SAMPLING_RATE_COLUMN", "sample_rate"))

    parser.add_argument("--generated-split-dir", type=str, default=env_path("GENERATED_SPLIT_DIR"))

    # Model
    parser.add_argument("--model-name-or-path", type=str, default=env_str("MODEL_NAME_OR_PATH", "openai/whisper-small"))
    parser.add_argument("--language", type=str, default=env_str("LANGUAGE", "amharic"))
    parser.add_argument("--task", type=str, default=env_str("TASK", "transcribe"))
    parser.add_argument("--cache-dir", type=str, default=env_path("CACHE_DIR"))
    parser.add_argument("--token", type=str, default=env_str("HF_TOKEN"))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=env_bool("TRUST_REMOTE_CODE", False))

    # Audio/dataset filtering
    parser.add_argument("--target-sampling-rate", type=int, default=env_int("TARGET_SAMPLING_RATE", 16000))
    parser.add_argument("--min-duration-s", type=float, default=env_float("MIN_DURATION_S", 0.5))
    parser.add_argument("--max-duration-s", type=float, default=env_float("MAX_DURATION_S", 30.0))
    parser.add_argument("--eval-size", type=float, default=env_float("EVAL_SIZE", 0.1))
    parser.add_argument("--test-size", type=float, default=env_float("TEST_SIZE", 0.0))

    parser.add_argument("--validate-audio-files", action=argparse.BooleanOptionalAction, default=env_bool("VALIDATE_AUDIO_FILES", True))
    parser.add_argument("--recompute-durations", action=argparse.BooleanOptionalAction, default=env_bool("RECOMPUTE_DURATIONS", False))
    parser.add_argument("--strict-transcript-chars", action=argparse.BooleanOptionalAction, default=env_bool("STRICT_TRANSCRIPT_CHARS", False))
    parser.add_argument("--do-lower-case", action=argparse.BooleanOptionalAction, default=env_bool("DO_LOWER_CASE", False))

    # Preprocessing
    parser.add_argument("--preprocessing-num-workers", type=int, default=env_int("PREPROCESSING_NUM_WORKERS", 4))
    parser.add_argument("--max-label-length", type=int, default=env_int("MAX_LABEL_LENGTH", None))
    parser.add_argument("--max-train-samples", type=int, default=env_int("MAX_TRAIN_SAMPLES", None))
    parser.add_argument("--max-eval-samples", type=int, default=env_int("MAX_EVAL_SAMPLES", None))
    parser.add_argument("--max-test-samples", type=int, default=env_int("MAX_TEST_SAMPLES", None))

    # Training behavior
    parser.add_argument("--output-dir", type=str, default=env_path("OUTPUT_DIR", "./whisper_amharic_outputs"))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--log-level", type=str, default=env_str("LOG_LEVEL", "INFO"))

    parser.add_argument("--apply-spec-augment", action=argparse.BooleanOptionalAction, default=env_bool("APPLY_SPEC_AUGMENT", False))
    parser.add_argument("--freeze-feature-encoder", action=argparse.BooleanOptionalAction, default=env_bool("FREEZE_FEATURE_ENCODER", False))
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=env_bool("FREEZE_ENCODER", False))
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=env_bool("GRADIENT_CHECKPOINTING", False))

    parser.add_argument("--sanity-overfit", action=argparse.BooleanOptionalAction, default=env_bool("SANITY_OVERFIT", False))
    parser.add_argument("--sanity-num-examples", type=int, default=env_int("SANITY_NUM_EXAMPLES", 2))

    parser.add_argument("--early-stopping-patience", type=int, default=env_int("EARLY_STOPPING_PATIENCE", 5))
    parser.add_argument("--log-prediction-examples", action=argparse.BooleanOptionalAction, default=env_bool("LOG_PREDICTION_EXAMPLES", True))
    parser.add_argument("--max-preview-examples", type=int, default=env_int("MAX_PREVIEW_EXAMPLES", 5))

    parser.add_argument("--do-predict-test", action=argparse.BooleanOptionalAction, default=env_bool("DO_PREDICT_TEST", False))

    # Seq2SeqTrainingArguments essentials
    parser.add_argument(
        "--do-train",
        action=argparse.BooleanOptionalAction,
        default=env_bool("DO_TRAIN", True),
        help="Whether to run training."
    )

    parser.add_argument(
        "--do-eval",
        action=argparse.BooleanOptionalAction,
        default=env_bool("DO_EVAL", True),
        help="Whether to run evaluation."
    )

    parser.add_argument(
        "--predict-with-generate",
        action=argparse.BooleanOptionalAction,
        default=env_bool("PREDICT_WITH_GENERATE", True),
        help="Whether to use generation during evaluation for WER/CER computation."
    )
    parser.add_argument("--num-train-epochs", type=float, default=env_float("NUM_TRAIN_EPOCHS", 10.0))
    parser.add_argument("--per-device-train-batch-size", type=int, default=env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 4))
    parser.add_argument("--per-device-eval-batch-size", type=int, default=env_int("PER_DEVICE_EVAL_BATCH_SIZE", 4))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=env_int("GRADIENT_ACCUMULATION_STEPS", 4))
    parser.add_argument("--learning-rate", type=float, default=env_float("LEARNING_RATE", 1e-5))
    parser.add_argument("--weight-decay", type=float, default=env_float("WEIGHT_DECAY", 0.01))
    parser.add_argument("--warmup-steps", type=int, default=env_int("WARMUP_STEPS", 500))
    parser.add_argument("--logging-steps", type=int, default=env_int("LOGGING_STEPS", 25))
    parser.add_argument("--eval-steps", type=int, default=env_int("EVAL_STEPS", 500))
    parser.add_argument("--save-steps", type=int, default=env_int("SAVE_STEPS", 500))
    parser.add_argument("--save-total-limit", type=int, default=env_int("SAVE_TOTAL_LIMIT", 3))
    parser.add_argument("--generation-max-length", type=int, default=env_int("GENERATION_MAX_LENGTH", 225))
    parser.add_argument("--generation-num-beams", type=int, default=env_int("GENERATION_NUM_BEAMS", 1))
    parser.add_argument("--dataloader-num-workers", type=int, default=env_int("DATALOADER_NUM_WORKERS", 4))
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=env_bool("FP16", torch.cuda.is_available()))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=env_bool("BF16", False))

    parser.add_argument("--resume-from-checkpoint", type=str, default=env_path("RESUME_FROM_CHECKPOINT"))

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.full_csv and not (args.train_csv and args.eval_csv):
        raise ValueError("Provide either --full-csv or both --train-csv and --eval-csv")

    if not args.output_dir:
        raise ValueError("--output-dir is required")

    if args.min_duration_s < 0:
        raise ValueError("--min-duration-s must be non-negative")
    if args.max_duration_s <= args.min_duration_s:
        raise ValueError("--max-duration-s must be greater than --min-duration-s")

    if args.per_device_train_batch_size <= 0:
        raise ValueError("--per-device-train-batch-size must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")

def load_json_config(config_path: str) -> dict:
    path = Path(config_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a JSON object")

    return config


def apply_config_to_parser_defaults(parser: argparse.ArgumentParser, config: dict) -> None:
    """
    Allows:
        python whisper_main.py whisper_config.json

    The JSON keys must match argparse destination names, e.g.
        train_csv, eval_csv, output_dir, learning_rate
    """
    valid_dests = {
        action.dest
        for action in parser._actions
        if action.dest != "help"
    }

    unknown_keys = sorted(set(config.keys()) - valid_dests)

    if unknown_keys:
        raise ValueError(
            "Unknown config keys found:\n"
            + "\n".join(f"  - {key}" for key in unknown_keys)
        )

    parser.set_defaults(**config)

def main() -> None:
    parser = build_arg_parser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file."
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config = load_json_config(sys.argv[1])
        apply_config_to_parser_defaults(parser, config)
        args = parser.parse_args([])
    else:
        known_args, remaining_args = parser.parse_known_args()

        if known_args.config:
            config = load_json_config(known_args.config)
            apply_config_to_parser_defaults(parser, config)
            args = parser.parse_args(remaining_args)
        else:
            args = parser.parse_args()

    setup_logging(args.log_level)
    validate_args(args)
    set_seed(args.seed)

    if args.log_level.upper() == "DEBUG":
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_warning()

    logger.info("Starting Whisper Amharic ASR fine-tuning")
    logger.info("Arguments: %s", vars(args))

    raw_datasets = load_local_manifest_datasets(
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        test_csv=args.test_csv,
        full_csv=args.full_csv,
        split_column=args.split_column,
        audio_path_column=args.audio_path_column,
        text_column=args.text_column,
        duration_column=args.duration_column,
        sampling_rate_column=args.sampling_rate_column,
        target_sampling_rate=args.target_sampling_rate,
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
        strict_transcript_chars=args.strict_transcript_chars,
        do_lower_case=args.do_lower_case,
        validate_audio_files=args.validate_audio_files,
        recompute_durations=args.recompute_durations,
        eval_size=args.eval_size,
        test_size=args.test_size,
        seed=args.seed,
        generated_split_dir=args.generated_split_dir,
    )

    if args.sanity_overfit:
        raw_datasets = make_sanity_overfit_dataset(
            raw_datasets,
            num_examples=args.sanity_num_examples,
        )

    processor = create_processor(
        args.model_name_or_path,
        language=args.language,
        task=args.task,
        cache_dir=args.cache_dir,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )

    forward_attention_mask = bool(args.apply_spec_augment)

    vectorized_datasets = preprocess_dataset(
        raw_datasets,
        processor=processor,
        target_sampling_rate=args.target_sampling_rate,
        preprocessing_num_workers=args.preprocessing_num_workers,
        forward_attention_mask=forward_attention_mask,
        max_label_length=args.max_label_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_test_samples=args.max_test_samples,
    )

    model = create_model(
        args.model_name_or_path,
        processor=processor,
        language=args.language,
        task=args.task,
        cache_dir=args.cache_dir,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        apply_spec_augment=args.apply_spec_augment,
        freeze_feature_encoder=args.freeze_feature_encoder,
        freeze_encoder=args.freeze_encoder,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
    )

    save_training_config(vars(args), args.output_dir)

    trainer = create_trainer(
        model=model,
        processor=processor,
        training_args=training_args,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["eval"],
        forward_attention_mask=forward_attention_mask,
        early_stopping_patience=args.early_stopping_patience,
        log_prediction_examples=args.log_prediction_examples,
        max_preview_examples=args.max_preview_examples,
    )

    test_dataset = vectorized_datasets["test"] if "test" in vectorized_datasets else None

    results = train_and_evaluate(
        trainer=trainer,
        processor=processor,
        output_dir=args.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict_test=args.do_predict_test,
        test_dataset=test_dataset,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    logger.info("Training complete. Results: %s", results)

if __name__ == "__main__":
    main()