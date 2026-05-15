from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import librosa
import torch
from transformers import WhisperProcessor

from Logger import LOGGER

logger = LOGGER.getChild("collator")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Dynamic padding collator for Whisper seq2seq ASR fine-tuning.

    This version supports two modes:

    1. Precomputed mode:
       examples already contain input_features and labels.

    2. Lazy mode:
       examples contain audio_path and text/transcript.
       Audio is loaded and converted to Whisper log-Mel features per batch.

    Lazy mode avoids storing all log-Mel features for the full dataset and
    prevents RAM/cache blow-up on large local ASR datasets.
    """

    processor: WhisperProcessor
    decoder_start_token_id: int
    forward_attention_mask: bool = True
    target_sampling_rate: int = 16000
    audio_path_column: str = "audio_path"
    text_column: str = "text"

    def __post_init__(self) -> None:
        if not isinstance(self.processor, WhisperProcessor):
            raise TypeError("processor must be an instance of WhisperProcessor")

        if not isinstance(self.decoder_start_token_id, int):
            raise TypeError("decoder_start_token_id must be an integer")

    def load_audio(self, audio_path: str):
        speech_array, _ = librosa.load(
            audio_path,
            sr=self.target_sampling_rate,
            mono=True,
        )

        if speech_array is None or len(speech_array) == 0:
            raise ValueError(f"Decoded audio is empty: {audio_path}")

        return speech_array

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not isinstance(features, list) or not features:
            raise ValueError("features must be a non-empty list")

        model_input_name = self.processor.model_input_names[0]

        input_features = []
        label_features = []

        for feature in features:
            # -------------------------
            # Encoder input preparation
            # -------------------------
            if model_input_name in feature:
                # Precomputed mode
                item = {
                    model_input_name: feature[model_input_name]
                }

                if "attention_mask" in feature:
                    item["attention_mask"] = feature["attention_mask"]

            else:
                # Lazy mode
                if self.audio_path_column not in feature:
                    raise KeyError(
                        f"Expected either '{model_input_name}' or "
                        f"'{self.audio_path_column}' in dataset example."
                    )

                audio_path = feature[self.audio_path_column]
                speech_array = self.load_audio(audio_path)

                inputs = self.processor.feature_extractor(
                    speech_array,
                    sampling_rate=self.target_sampling_rate,
                    return_attention_mask=True,
                )

                item = {
                    model_input_name: inputs[model_input_name][0],
                    "attention_mask": inputs["attention_mask"][0],
                }

            input_features.append(item)

            # -------------------------
            # Decoder label preparation
            # -------------------------
            if "labels" in feature:
                label_ids = feature["labels"]
            else:
                if self.text_column in feature:
                    text = feature[self.text_column]
                elif "transcript" in feature:
                    text = feature["transcript"]
                else:
                    raise KeyError(
                        f"Expected 'labels', '{self.text_column}', or 'transcript' "
                        "in dataset example."
                    )

                label_ids = self.processor.tokenizer(str(text)).input_ids

            label_features.append({"input_ids": label_ids})

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        # Remove BOS token if tokenizer added it to every label sequence.
        if (
            labels.shape[1] > 0
            and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels

        return dict(batch)