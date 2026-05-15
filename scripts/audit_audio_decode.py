import os
import pandas as pd
import librosa
from tqdm import tqdm

CSV_PATHS = [
    "/mnt/data-disk/home/Ibsa-projects/ASR-Folder/ASR_experiment/Amharic_ASR_whisper/Dataset/Training_Dataset/merged_train_dataset.csv",
    "/mnt/data-disk/home/Ibsa-projects/ASR-Folder/ASR_experiment/Amharic_ASR_whisper/Dataset/Training_Dataset/merged_validation_dataset.csv",
    "/mnt/data-disk/home/Ibsa-projects/ASR-Folder/ASR_experiment/Amharic_ASR_whisper/Dataset/Training_Dataset/merged_test_dataset.csv",
]

TARGET_SR = 16000

for csv_path in CSV_PATHS:
    print(f"\nAuditing: {csv_path}")
    df = pd.read_csv(csv_path)

    bad_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = str(row["audio_path"])

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(audio_path)

            audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

            if audio is None or len(audio) == 0:
                raise ValueError("empty decoded audio")

        except Exception as exc:
            bad = row.to_dict()
            bad["row_index"] = idx
            bad["decode_error"] = f"{type(exc).__name__}: {exc}"
            bad_rows.append(bad)

    if bad_rows:
        bad_df = pd.DataFrame(bad_rows)
        out_path = csv_path.replace(".csv", ".decode_failed.csv")
        bad_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Found {len(bad_rows)} bad audio files. Saved: {out_path}")
    else:
        print("No decode failures found.")