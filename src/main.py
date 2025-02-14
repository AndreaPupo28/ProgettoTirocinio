import torch
import os
import random
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from data_loader import load_dataset
from model import BertClassifier
from train import train
from evaluation import evaluate_model
from predict import predict_next_log

if __name__ == "__main__":
    model_name = "prajjwal1/bert-medium"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    dataset_path = "/kaggle/working/ProgettoTirocinio/dataset/BPIC15_1.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Errore: Il file CSV '{dataset_path}' non esiste!")

    dataset = load_dataset(dataset_path, tokenizer)
    output_size = dataset.num_classes

    # Divisione in training (80%) e test (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    model = BertClassifier(model_name, output_size).to(device)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    if not os.path.exists("/kaggle/working/modello_addestrato.pth"):
        print("\nAvvio dell'addestramento...")
        model = train(model, train_loader, optimizer, 10, criterion, device)
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/modello_addestrato.pth")
        print("\nModello addestrato e salvato con successo.")
    else:
        print("\nCaricamento del modello già addestrato...")
        model.load_state_dict(torch.load("/kaggle/working/modello_addestrato.pth"))

    # Valutazione del modello dopo il training
    print("\nValutazione del modello sul test set...")
    evaluate_model(model, test_loader, criterion, device)

    model.eval()
    for case_sequence, _ in dataset.data:
        generated_sequence = [case_sequence[0]]

        print("\n--------------------------------------")
        print(f"Inizio della generazione per il case: {' → '.join(generated_sequence)}")
        print("--------------------------------------\n")

        while True:
            input_text = " ".join(generated_sequence)  # Converte la lista in stringa
            predicted_next, _ = predict_next_log(model, tokenizer, input_text, dataset.label_map, device)
            
            print(f"Traccia generata finora: {' → '.join(generated_sequence)}")
            print(f"Prossima attività predetta: {predicted_next}\n")
 
            if predicted_next is None or predicted_next in generated_sequence:
                print("Fine della traccia per questo case: nessuna nuova attività da predire.")
                break

            generated_sequence.append(predicted_next)

