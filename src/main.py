import torch
import os
import random
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from data_loader import load_dataset
from model import BertClassifier
from predict import predict_next_log_with_constraints
from train import train
from evaluation import evaluate_model

if __name__ == "__main__":
    model_name = "prajjwal1/bert-medium"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    dataset_path = "/kaggle/working/ProgettoTirocinio/dataset/BPIC15_1.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Errore: Il file CSV '{dataset_path}' non esiste!")

    df = pd.read_csv(dataset_path, low_memory=False)

    # Verifica il nome corretto della colonna dei case (assumendo che sia 'case')
    case_column = "case"
    activity_column = "activity"

    if case_column not in df.columns or activity_column not in df.columns:
        raise KeyError("Errore: Il file CSV non contiene le colonne richieste.")

    # Raggruppa le attività per ogni case (dizionario)
    grouped_cases = df.groupby(case_column)[activity_column].apply(list).to_dict()

    # Creazione del modello
    model = BertClassifier(model_name, output_size=len(set(df[activity_column]))).to(device)

    # Caricamento del modello addestrato
    if not os.path.exists("/kaggle/working/modello_addestrato.pth"):
        print("\nAvvio dell'addestramento...")
        dataset = load_dataset(dataset_path, tokenizer)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        model = train(model, train_loader, optimizer, 10, criterion, device)
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "/kaggle/working/modello_addestrato.pth")
        print("\nModello addestrato e salvato con successo.")
    else:
        print("\nCaricamento del modello già addestrato...")
        model.load_state_dict(torch.load("/kaggle/working/modello_addestrato.pth"))
        model.eval() # impostato in modalità valutazione

    print("\nValutazione del modello sul test set...")
    dataset = load_dataset(dataset_path, tokenizer)
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    evaluate_model(model, test_loader, criterion, device)

    for case_id, case_sequence in grouped_cases.items():
        print("\n--------------------------------------")
        print(f"Inizio della generazione per il case {case_id}")
        print("--------------------------------------\n")

        # Inizializza la sequenza con la prima attività
        generated_sequence = [case_sequence[0]]

        while True:
            input_text = " ".join(generated_sequence)

            predicted_sequences = predict_next_log_with_constraints(
                model, tokenizer, input_text, dataset.label_map, device
            )

            if not predicted_sequences or not predicted_sequences[0]:  # Se non ci sono sequenze generate
                print(f"Fine della traccia per il case {case_id}: nessuna nuova attività da predire.")
                break

            predicted_next_name, predicted_next_prob = predicted_sequences[0][0]  # Estrai nome e probabilità

            print(f"Prossima attività predetta: {predicted_next_name} con probabilità {predicted_next_prob:.4f}\n")
            generated_sequence.append(predicted_next_name)


