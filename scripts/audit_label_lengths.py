import pandas as pd
from transformers import WhisperProcessor
from tqdm import tqdm

CSV_PATHS = [
    "/mnt/data-disk/home/Ibsa-projects/ASR-Folder/ASR_experiment/Amharic_ASR_whisper/Dataset/Training_Dataset/merged_train_dataset.csv",
    "/mnt/data-disk/home/Ibsa-projects/ASR-Folder/ASR_experiment/Amharic_ASR_whisper/Dataset/Training_Dataset/merged_validation_dataset.csv",
    "/mnt/data-disk/home/Ibsa-projects/ASR-Folder/ASR_experiment/Amharic_ASR_whisper/Dataset/Training_Dataset/merged_test_dataset.csv",
]

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="amharic",
    task="transcribe",
)

MAX_LABEL_LENGTH = 440

for csv_path in CSV_PATHS:
    print(f"\nAuditing: {csv_path}")
    df = pd.read_csv(csv_path)

    rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row["transcript"])
        token_ids = processor.tokenizer(text).input_ids
        length = len(token_ids)

        if length > MAX_LABEL_LENGTH:
            item = row.to_dict()
            item["row_index"] = idx
            item["label_length"] = length
            rows.append(item)

    print(f"Rows longer than {MAX_LABEL_LENGTH} tokens: {len(rows)}")

    if rows:
        out_path = csv_path.replace(".csv", f".label_too_long_gt_{MAX_LABEL_LENGTH}.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {out_path}")