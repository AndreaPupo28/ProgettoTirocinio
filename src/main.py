import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer
from src.bert_output_classification_head import BertOutputClassificationHead
from src.trainer import train


class LogDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f" Errore: Il file CSV '{file_path}' non esiste!")

        self.data = pd.read_csv(file_path, nrows=1000)  # Carica solo le prime 1000 righe
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Scegli la colonna corretta per i label
        label_column = "concept:name"  # Cambia se necessario

        if label_column not in self.data.columns:
            raise ValueError(f" Errore: La colonna '{label_column}' non esiste nel CSV!")

        # Creiamo una mappa che assegna numeri alle classi
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data[label_column].dropna().unique()))}

        # Applichiamo la mappatura ai dati
        self.data["label"] = self.data[label_column].map(self.label_map)

        # Definiamo il numero totale di classi
        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        log_entry = str(self.data.iloc[idx]["activity"])  # Testo del log

        label = self.data.iloc[idx]["label"]
        if pd.isna(label):  # Controlla i NaN nei label
            raise ValueError(f" Errore: Label NaN alla riga {idx} del CSV!")

        label = int(label)  # Converte il label in intero
        # Tokenizza il log con BERT
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
            "label": torch.tensor(label, dtype=torch.long),  # Ora è un numero intero
        }


if __name__ == "__main__":
    model_name = "prajjwal1/bert-medium"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Device usato per il training: {device}")
    learning_rate = 1e-5
    epochs = 3

    # Caricamento del Tokenizer e del Modello BERT
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    model = AutoModel.from_pretrained(model_name)

    dataset_path = "dataset/BPIC15_1.csv"

    # Creiamo il dataset PRIMA di definire output_size!
    train_dataset = LogDataset(dataset_path, tokenizer, max_length=128)

    # Definiamo il numero corretto di classi
    output_size = train_dataset.num_classes  # Ora funziona!
    print(f" Numero di classi: {output_size}")
    print(f" Classi trovate: {train_dataset.label_map}")

    # Aggiunge la testa di classificazione (Linear Layer)
    model = BertOutputClassificationHead(model, output_size).to(device)
    import os

    if os.path.exists("modello_addestrato.pth"):
        model.load_state_dict(torch.load("modello_addestrato.pth"))
        model.eval()  # Imposta il modello in modalità inferenza
        print("Modello pre-addestrato caricato con successo!")
    else:
        print("Nessun modello salvato trovato. Il training partirà da zero.")

    # Creiamo i DataLoader per il Training e il Test
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

    # Configurazione dell'addestramento
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Avvia l'addestramento solo se il modello non è già stato caricato
    if not os.path.exists("modello_addestrato.pth"):
        model = train(model, train_loader, test_loader, optimizer, epochs, criterion, device)
        torch.save(model.state_dict(), "modello_addestrato.pth")
        print("Modello salvato con successo!")
    else:
        print("Modello già addestrato. Salto il training.")
