import pandas as pd
from torch.utils.data import Dataset
import torch


class LogDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = pd.read_csv(file_path, low_memory=False)
        print(f"Dataset caricato: {len(self.data)} righe trovate.")

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Creare un dataset basato su sequenze incrementali per ogni case
        self.data = self.data.sort_values(by=["case", "timestamp"])  # Ordina per case e tempo
        grouped = self.data.groupby("case")["activity"].apply(list)

        sequences = []
        for case in grouped:
            for i in range(1, len(case)):
                sequences.append((case[:i], case[i]))  # X = [A1, A2, ..., Ai], Y = Ai+1

        self.data = sequences

        # Creare la mappatura delle attività a indici numerici
        unique_activities = sorted(set(a for seq in self.data for a in seq[0] + [seq[1]]))
        self.label_map = {label: idx for idx, label in enumerate(unique_activities)}
        self.num_classes = len(self.label_map)

        #print(f"Classi trovate: {self.label_map}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, next_activity = self.data[idx]

        # Convertire la sequenza in stringa unica per la tokenizzazione
        input_text = " ".join(sequence)
        next_label = self.label_map[next_activity]

        inputs = self.tokenizer(
            input_text,
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
