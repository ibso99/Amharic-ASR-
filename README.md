# Amharic-ASR-
# Amharic ASR Dataset EDA

This project performs exploratory data analysis on a merged Amharic ASR training dataset created from Waxal and Librispeech-style Amharic speech data.

The goal was to validate the dataset before fine-tuning a Whisper-based speech recognition model.

## Dataset

- Total samples: 39,022
- Split analyzed: Train
- Sources: Waxal + Librispeech-style Amharic data
- Target model: Whisper
- Target sample rate: 16 kHz

## Key Findings

- 99.9% of audio files are within the 0.5–30 second target range
- Median duration: 17.38 seconds
- Mean duration: 17.64 seconds
- 100% of inspected files use the target 16 kHz sample rate
- 0 empty transcripts
- 98.5% pure Amharic transcripts
- 566 unique speakers
- 515 duplicate transcripts require leakage checks
- 20 files are longer than 30 seconds
- 1,000 samples have missing language, gender, and speaker metadata

## Conclusion

The dataset is suitable for Whisper fine-tuning after minor preprocessing. Recommended steps include duration filtering, mixed-script transcript review, duplicate transcript inspection, speaker-aware splitting, and final audio quality validation.