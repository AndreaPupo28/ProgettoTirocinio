import pandas as pd
from torch.utils.data import Dataset
import torch
from activity import ActivityPrediction


class LogDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        # Caricamento del dataset
        self.data = pd.read_csv(file_path, low_memory=False)
        print(f"Dataset caricato: {len(self.data)} righe trovate.")

        self.tokenizer = tokenizer
        self.max_length = max_length  # lunghezza massima della sequenza dopo la tokenizzazione

        # Ordina il dataset per case e timestamp
        self.data = self.data.sort_values(by=["case", "timestamp"])
        grouped = self.data.groupby("case")["activity"].apply(list)  # Raggruppa le attività per case

        # Conserva le tracce originali per eventuali valutazioni successive
        self.traces = grouped.tolist()

        sequences = []
        # Genera sequenze incrementali per ogni traccia
        for trace in grouped:
            for i in range(1, len(trace)):
                sequences.append((trace[:i], ActivityPrediction(trace[i], 1.0)))
            # Aggiunge la sequenza finale che termina con "END OF SEQUENCE"
            sequences.append((trace, ActivityPrediction("END OF SEQUENCE", 1.0)))

        self.data = sequences

        # Estrae tutte le attività uniche dal dataset, incluse quelle in input e quelle predette,
        # in modo da creare il dizionario label_map
        unique_activities = sorted(
            set(
                a.name if isinstance(a, ActivityPrediction) else a
                for seq in self.data for a in seq[0] + [seq[1]]
            )
        )
        self.label_map = {label: idx for idx, label in enumerate(unique_activities)}
        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, next_activity = self.data[idx]

        # Converte la sequenza in stringa unica per la tokenizzazione
        input_text = " ".join(sequence)
        if isinstance(next_activity, ActivityPrediction):
            next_label = self.label_map[next_activity.name]
        else:
            next_label = self.label_map[next_activity]

        inputs = self.tokenizer(
            input_text, # da stringa a numeri
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True # se la sequenza è più lunga la tronca
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0), # rimuove la dimensione extra aggiunta dal tokenizer
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(next_label, dtype=torch.long) # converte next_label in un tensore di tipo long
        }

# wrapper
def load_dataset(file_path, tokenizer, max_length=128):
    return LogDataset(file_path, tokenizer, max_length)
