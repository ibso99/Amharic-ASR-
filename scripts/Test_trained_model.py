"""
Amharic Speech Recognition using Fine-tuned Whisper
==================================================
Transcribe Amharic audio files using a fine-tuned Whisper model
"""

import os
import re
from pathlib import Path
import torch
import librosa
import pandas as pd
from IPython.display import Audio, display
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm.auto import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for ASR inference"""
    # Paths
    MODEL_DIR = "/mnt/data-disk/home/Ibsa-projects/ASR-Folder/ASR_experiment/Amharic_ASR_whisper/Outputs/whisper_small_amharic_normalized_v2/final_model"
    TEST_CSV = "/mnt/data-disk/home/Ibsa-projects/ASR-Folder/ASR_experiment/Amharic_ASR_whisper/Dataset/Training_Dataset/merged_test_dataset.csv"
    
    # Data columns
    AUDIO_COL = "audio_path"
    TEXT_COL = "transcript"
    
    # Audio settings
    TARGET_SR = 16000
    LANGUAGE = "amharic"
    TASK = "transcribe"
    
    # Generation settings
    MAX_NEW_TOKENS = 444
    NUM_BEAMS = 1
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

class AmharicNormalizer:
    """Amharic text normalization for ASR"""
    
    # Character mappings for similar sounds
    CHAR_MAP = {
        # ሐ family → ሀ
        'ሐ': 'ሀ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሃ', 'ሔ': 'ሄ', 'ሕ': 'ህ', 'ሖ': 'ሆ',
        # ኀ family → ሀ
        'ኀ': 'ሀ', 'ኁ': 'ሁ', 'ኂ': 'ሂ', 'ኃ': 'ሃ', 'ኄ': 'ሄ', 'ኅ': 'ህ', 'ኆ': 'ሆ',
        # ዐ family → አ
        'ዐ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዓ': 'አ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ',
        # ጸ family → ፀ
        'ጸ': 'ፀ', 'ጹ': 'ፁ', 'ጺ': 'ፂ', 'ጻ': 'ፃ', 'ጼ': 'ፄ', 'ጽ': 'ፅ', 'ጾ': 'ፆ',
    }
    
    # Punctuation to remove
    PUNCTUATION = r"[።፣፤፥፦፧፨.,!?;:\"'()\[\]{}]"
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalize Amharic text for evaluation"""
        text = str(text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Map similar characters
        for src, tgt in cls.CHAR_MAP.items():
            text = text.replace(src, tgt)
        
        # Remove punctuation
        text = re.sub(cls.PUNCTUATION, "", text)
        
        # Final whitespace cleanup
        text = re.sub(r"\s+", " ", text).strip()
        
        return text


# ============================================================================
# MODEL LOADER
# ============================================================================

class WhisperASR:
    """Whisper ASR model wrapper for Amharic transcription"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load processor and model from fine-tuned checkpoint"""
        print(f"Loading model from: {self.config.MODEL_DIR}")
        
        # Load processor
        self.processor = WhisperProcessor.from_pretrained(
            self.config.MODEL_DIR,
            language=self.config.LANGUAGE,
            task=self.config.TASK,
        )
        
        # Load model
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config.MODEL_DIR
        )
        self.model.to(self.config.DEVICE)
        self.model.eval()
        
        # Configure generation
        self.model.generation_config.forced_decoder_ids = (
            self.processor.get_decoder_prompt_ids(
                language=self.config.LANGUAGE,
                task=self.config.TASK
            )
        )
        self.model.generation_config.language = self.config.LANGUAGE
        self.model.generation_config.task = self.config.TASK
        self.model.generation_config.return_timestamps = False
        
        print(f"✅ Model loaded on: {self.config.DEVICE}")
    
    def transcribe(self, audio_path: str, num_beams: int = None) -> str:
        """Transcribe a single audio file"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        
        # Load audio
        speech, _ = librosa.load(audio_path, sr=self.config.TARGET_SR, mono=True)
        
        # Extract features
        inputs = self.processor.feature_extractor(
            speech,
            sampling_rate=self.config.TARGET_SR,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Move to device
        input_features = inputs.input_features.to(self.config.DEVICE)
        attention_mask = inputs.attention_mask.to(self.config.DEVICE) if hasattr(inputs, "attention_mask") else None
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                num_beams=num_beams or self.config.NUM_BEAMS,
                do_sample=False,
                language=self.config.LANGUAGE,
                task=self.config.TASK,
                return_timestamps=False,
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return transcription
    
    def transcribe_batch(self, audio_paths: list, num_beams: int = None) -> list:
        """Transcribe multiple audio files with progress bar"""
        results = []
        for path in tqdm(audio_paths, desc="Transcribing"):
            try:
                text = self.transcribe(path, num_beams)
                results.append({"path": path, "transcription": text, "error": None})
            except Exception as e:
                results.append({"path": path, "transcription": "", "error": str(e)})
        return results


# ============================================================================
# DATA LOADER
# ============================================================================

class TestDataLoader:
    """Load and manage test dataset"""
    
    def __init__(self, config: Config):
        self.config = config
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load test CSV"""
        if not os.path.exists(self.config.TEST_CSV):
            raise FileNotFoundError(f"Test CSV not found: {self.config.TEST_CSV}")
        
        self.df = pd.read_csv(self.config.TEST_CSV)
        print(f"✅ Loaded {len(self.df)} test samples")
    
    def get_random_sample(self):
        """Get random audio path and reference text"""
        import random
        idx = random.randint(0, len(self.df) - 1)
        return {
            "audio_path": self.df.iloc[idx][self.config.AUDIO_COL],
            "reference": self.df.iloc[idx][self.config.TEXT_COL],
            "index": idx
        }
    
    def get_all_samples(self):
        """Get all audio paths and references"""
        return [
            {"audio_path": row[self.config.AUDIO_COL], "reference": row[self.config.TEXT_COL]}
            for _, row in self.df.iterrows()
        ]


# ============================================================================
# EVALUATION
# ============================================================================

class Evaluator:
    """Evaluate ASR model performance"""
    
    def __init__(self, asr_model: WhisperASR, normalizer: AmharicNormalizer):
        self.asr = asr_model
        self.normalizer = normalizer
        self.wer = evaluate.load("wer")
        self.cer = evaluate.load("cer")
    
    def compute_metrics(self, references: list, predictions: list) -> dict:
        """Compute WER and CER metrics"""
        # Normalize texts
        ref_norm = [self.normalizer.normalize(r) for r in references]
        pred_norm = [self.normalizer.normalize(p) for p in predictions]
        
        return {
            "wer": self.wer.compute(predictions=pred_norm, references=ref_norm),
            "cer": self.cer.compute(predictions=pred_norm, references=ref_norm),
        }
    
    def evaluate_test_set(self, test_data: TestDataLoader, num_beams: int = None) -> dict:
        """Evaluate on entire test set"""
        samples = test_data.get_all_samples()
        
        references = []
        predictions = []
        
        for sample in tqdm(samples, desc="Evaluating"):
            try:
                pred = self.asr.transcribe(sample["audio_path"], num_beams)
                references.append(sample["reference"])
                predictions.append(pred)
            except Exception as e:
                print(f"Error: {sample['audio_path']} - {e}")
        
        return self.compute_metrics(references, predictions)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution workflow"""
    print("=" * 60)
    print("🎙️  Amharic ASR Transcription System")
    print("=" * 60)
    
    # Initialize components
    config = Config()
    normalizer = AmharicNormalizer()
    asr_model = WhisperASR(config)
    test_data = TestDataLoader(config)
    evaluator = Evaluator(asr_model, normalizer)
    
    # Test with random sample
    print("\n" + "=" * 60)
    print("📝 Testing Random Sample")
    print("=" * 60)
    
    sample = test_data.get_random_sample()
    
    print(f"\n📁 Audio: {sample['audio_path']}")
    print(f"📖 Reference: {normalizer.normalize(sample['reference'])}")
    
    # Display audio
    display(Audio(sample['audio_path'], rate=config.TARGET_SR))
    
    # Transcribe
    prediction = asr_model.transcribe(sample['audio_path'])
    print(f"🎙️ Prediction: {prediction}")
    
    # Optional: Evaluate on full test set
    # print("\n" + "=" * 60)
    # print("📊 Full Test Set Evaluation")
    # print("=" * 60)
    # metrics = evaluator.evaluate_test_set(test_data)
    # print(f"WER: {metrics['wer']:.2%}")
    # print(f"CER: {metrics['cer']:.2%}")


if __name__ == "__main__":
    main()