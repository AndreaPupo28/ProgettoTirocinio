import torch
import os
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from data_loader import load_dataset
from model import BertClassifier
from train import train
from evaluation import evaluate_model
from particle_filter import ParticleFilter
from log_similarity import evaluate_log_similarity

if __name__ == "__main__":
    model_name = "prajjwal1/bert-medium"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    dataset_path = "/kaggle/working/ProgettoTirocinio/dataset/BPIC15_1.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Errore: Il file CSV '{dataset_path}' non esiste!")

    df = pd.read_csv(dataset_path, low_memory=False)
    model = BertClassifier(model_name, output_size=len(set(df["activity"]))).to(device)

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
        model.eval()

    print("\nValutazione del modello sul test set...")
    dataset = load_dataset(dataset_path, tokenizer)
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    evaluate_model(model, test_loader, criterion, device)

    initial_activities = list(set(df["activity"].tolist()))[:5]  # prende le prime 5 attività uniche
    pf = ParticleFilter(model, tokenizer, dataset.label_map, device, num_particles=50)
    pf.initialize_particles(initial_activities)
    final_particles = pf.run(steps=1) #step: numero max di iterazioni per il PF per estendere le particelle

    similarity_score = evaluate_log_similarity(model, tokenizer, dataset, dataset.label_map, device)
    print(f"CFld Similarity (dopo generazione tracce): {similarity_score:.4f}")

    print("\nParticelle finali generate:")
    for particle in final_particles:
        print([act.name for act in particle])
