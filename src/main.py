import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer
from src.bert_output_classification_head import BertClassifier
from src.trainer import train, predict_next_log


class LogDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = pd.read_csv(file_path, low_memory=False)
        print(f" Dataset caricato: {len(self.data)} righe trovate.")

        self.tokenizer = tokenizer
        self.max_length = max_length

        #  Creiamo una colonna "next_log" con il valore della riga successiva
        self.data["next_log"] = self.data["activity"].shift(-1)  # Modificato per usare "activity"
        self.data.dropna(subset=["next_log"], inplace=True)  # Evita di cancellare tutto
        print(f" Dopo il dropna(), il dataset ha {len(self.data)} righe.")

        print(f" Dopo il dropna(), il dataset ha {len(self.data)} righe.")

        #  Mappa i log a indici numerici
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data["activity"].unique()))}
        self.data["label"] = self.data["activity"].map(self.label_map)

        #  Controllo se sono state trovate classi
        if len(self.label_map) == 0:
            raise ValueError(
                " Errore: Nessuna classe trovata nel dataset! La colonna 'concept:name' potrebbe essere vuota.")

        print(f" Classi trovate: {self.label_map}")

        self.data["next_label"] = self.data["next_log"].map(self.label_map)  # Mappa anche il log successivo

        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        log_entry = str(self.data.iloc[idx]["activity"])
        next_log = str(self.data.iloc[idx]["next_log"])  # Log successivo
        label = int(self.data.iloc[idx]["label"])
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
            "label": torch.tensor(label, dtype=torch.long),
            "next_label": torch.tensor(next_label, dtype=torch.long),  # Il log successivo come etichetta
        }

if __name__ == "__main__":
    model_name = "prajjwal1/bert-medium"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Device usato per il training: {device}")
    learning_rate = 1e-5
    epochs = 10

    # Caricamento del Tokenizer e del Modello BERT
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    model = AutoModel.from_pretrained(model_name)

    # Determina il percorso corretto in base all'ambiente (PC o Kaggle)
    local_dataset_path = r"C:\Users\Utente\OneDrive\Desktop\Università\Tirocinio\Progetto\dataset\BPIC15_1.csv"
    kaggle_dataset_path = "/kaggle/working/ProgettoTirocinio/dataset/BPIC15_1.csv"  # Modifica con il nome corretto del dataset su Kaggle

    if os.path.exists(local_dataset_path):
        dataset_path = local_dataset_path
    elif os.path.exists(kaggle_dataset_path):
        dataset_path = kaggle_dataset_path
    else:
        raise FileNotFoundError("Errore: Il file CSV non esiste né su PC né su Kaggle! Controlla il percorso.")

    print(f"✅ Dataset trovato in: {dataset_path}")

    #  Controllo se il file CSV esiste
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Errore: Il file CSV '{dataset_path}' non esiste! Controlla il percorso.")

    #  Controllo se il CSV ha dati
    df_test = pd.read_csv(dataset_path, nrows=5)  # Leggiamo solo 5 righe per testare
    print("Anteprima del dataset:", df_test.head())

    # 🔹 Controllo se la colonna "activity" esiste
    if "activity" not in df_test.columns:
        raise ValueError("Errore: La colonna 'activity' non esiste nel dataset! Controlla il file CSV.")

    # Creiamo il dataset PRIMA di definire output_size!
    # Usa solo una parte del dataset per il test
    df_full = pd.read_csv(dataset_path, low_memory=False)
    df_sample = df_full.sample(n=2000, random_state=42)  # Usa solo 2000 righe invece di tutto il dataset
    df_sample.to_csv("dataset_sample.csv", index=False)  # Salva il subset come nuovo CSV

    train_dataset = LogDataset("dataset_sample.csv", tokenizer, max_length=128)

    if len(train_dataset) == 0:
        raise ValueError("Errore: Il dataset è vuoto! Controlla il file CSV.")

    # Definiamo il numero corretto di classi
    output_size = train_dataset.num_classes
    print(f" Numero di classi: {output_size}")
    print(f" Classi trovate: {train_dataset.label_map}")

    #Controlla se il dataset è vuoto PRIMA di creare il modello!
    if output_size == 0:
        raise ValueError("Errore: Nessuna classe trovata nel dataset!")

    # Aggiunge la testa di classificazione (Linear Layer)
    model = BertClassifier(model, output_size).to(device)

    if os.path.exists("modello_addestrato.pth"):
        try:
            saved_model_state = torch.load("modello_addestrato.pth")
            saved_output_size = saved_model_state["output_layer.weight"].shape[0]  # Numero di classi salvate

            if saved_output_size != output_size:
                print(f"Vecchio modello eliminato (output_size {saved_output_size} ≠ {output_size}). Sarà riaddestrato da zero.")
                os.remove("modello_addestrato.pth")  # Elimina il vecchio modello
            else:
                print("Modello salvato compatibile. Procedo senza riaddestramento.")
                model.load_state_dict(saved_model_state)  # Carica il modello solo se è compatibile
        except Exception as e:
            print(f"Errore nel caricamento del modello salvato: {e}. Lo elimino e riaddestro.")
            os.remove("modello_addestrato.pth")
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

    # TEST: Predizione del prossimo log con Particle Filtering (SEMPRE ESEGUITA)
    model.eval()  # Mette il modello in modalità inferenza
    current_log = random.choice(train_dataset.data["activity"].tolist())

    predicted_next, probabilities = predict_next_log(model, tokenizer, current_log, train_dataset.label_map, device)

    print(f"Log attuale: {current_log}")
    print(f"Log successivo predetto: {predicted_next}")
