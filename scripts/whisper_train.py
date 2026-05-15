from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import evaluate
import numpy as np
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from whisper_collator_loader import DataCollatorSpeechSeq2SeqWithPadding
from whisper_utilities import normalize_amharic_for_asr
from Logger import LOGGER

logger = LOGGER.getChild("train")

def normalize_for_metrics(text: str) -> str:
    return " ".join(str(text).split()).strip()

class PredictionPreviewCallback(TrainerCallback):
    """
    Logs prediction/reference previews after evaluation when predict_with_generate=True.
    """
    def __init__(self, processor: WhisperProcessor, max_examples: int = 5):
        self.processor = processor
        self.max_examples = max_examples

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            logger.info("Evaluation metrics: %s", metrics)


def build_compute_metrics(
    processor: WhisperProcessor,
    *,
    log_prediction_examples: bool = True,
    max_preview_examples: int = 5,):

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = np.where(
            label_ids == -100,
            processor.tokenizer.pad_token_id,
            label_ids,
        )

        pred_str = processor.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
        )

        label_str = processor.tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True,
        )

        # pred_str = [normalize_for_metrics(x) for x in pred_str]
        # label_str = [normalize_for_metrics(x) for x in label_str]
        pred_str = [normalize_amharic_for_asr(text) for text in pred_str]
        label_str = [normalize_amharic_for_asr(text) for text in label_str]

        if log_prediction_examples:
            preview_n = min(max_preview_examples, len(pred_str))
            for i in range(preview_n):
                logger.info("EVAL PREVIEW %d | REF=%r | PRED=%r",i,label_str[i],pred_str[i],)

        wer_value = wer_metric.compute(
            predictions=pred_str,
            references=label_str,
        )

        cer_value = cer_metric.compute(
            predictions=pred_str,
            references=label_str,
        )

        return {
            "wer": float(wer_value),
            "cer": float(cer_value),
            "char_accuracy": float(max(0.0, min(1.0, 1.0 - cer_value))),
        }

    return compute_metrics

def create_trainer(
    *,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    training_args: Seq2SeqTrainingArguments,
    train_dataset,
    eval_dataset,
    forward_attention_mask: bool = True,
    early_stopping_patience: Optional[int] = None,
    log_prediction_examples: bool = True,
    max_preview_examples: int = 5,
) -> Seq2SeqTrainer:
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
        target_sampling_rate=16000,
        audio_path_column="audio_path",
        text_column="text",
    )

    callbacks = []

    if early_stopping_patience is not None and early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            )
        )

    callbacks.append(
        PredictionPreviewCallback(
            processor=processor,
            max_examples=max_preview_examples,
        )
    )

    compute_metrics = (
        build_compute_metrics(
            processor,
            log_prediction_examples=log_prediction_examples,
            max_preview_examples=max_preview_examples,
        )
        if training_args.predict_with_generate
        else None
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
    )

    return trainer

def save_training_config(payload: Dict[str, Any], output_dir: str) -> None:
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)

    config_path = path / "run_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    logger.info("Saved run configuration to %s", config_path)


def train_and_evaluate(
    *,
    trainer: Seq2SeqTrainer,
    processor: WhisperProcessor,
    output_dir: str,
    do_train: bool = True,
    do_eval: bool = True,
    do_predict_test: bool = False,
    test_dataset=None,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    if do_train:
        logger.info("*** Training Whisper model ***")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)

        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

        results["train"] = train_metrics

    if do_eval:
        logger.info("*** Evaluating Whisper model ***")
        eval_metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=trainer.args.generation_max_length,
            num_beams=trainer.args.generation_num_beams,
        )

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        results["eval"] = eval_metrics

    if do_predict_test and test_dataset is not None:
        logger.info("*** Predicting on test split ***")
        test_output = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=trainer.args.generation_max_length,
            num_beams=trainer.args.generation_num_beams,
        )

        trainer.log_metrics("test", test_output.metrics)
        trainer.save_metrics("test", test_output.metrics)
        results["test"] = test_output.metrics

    final_dir = Path(output_dir).expanduser().resolve() / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    logger.info("Saved final model and processor to %s", final_dir)

    return results