import pandas as pd
from torch.utils.data import Dataset
import torch


class LogDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = pd.read_csv(file_path, low_memory=False)
        print(f"Dataset caricato: {len(self.data)} righe trovate.")

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Creiamo una colonna "next_log" con il valore della riga successiva
        self.data["next_log"] = self.data["activity"].shift(-1)
        self.data.dropna(subset=["next_log"], inplace=True)

        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data["activity"].unique()))}
        self.data["label"] = self.data["activity"].map(self.label_map)
        self.data["next_label"] = self.data["next_log"].map(self.label_map)

        self.num_classes = len(self.label_map)

        print(f"Classi trovate: {self.label_map}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        log_entry = str(self.data.iloc[idx]["activity"])
        next_label = int(self.data.iloc[idx]["next_label"])

        inputs = self.tokenizer(
            log_entry,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(next_label, dtype=torch.long)
        }


def load_dataset(file_path, tokenizer, max_length=128):
    return LogDataset(file_path, tokenizer, max_length)
