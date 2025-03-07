import torch
import os
import pandas as pd
import argparse
import json
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from data_loader import load_dataset
from model import BertClassifier
from train import train
from evaluation import evaluate_model
from particle_filter import ParticleFilter
from log_similarity import evaluate_log_similarity

if __name__ == "__main__":
    # Parser per ottenere input dinamico tramite argparse
    parser = argparse.ArgumentParser(description='Esegui il modello con input dinamico')
    parser.add_argument('--activity', type=str, default="A", help='Attività iniziale per il filtro particellare')
    parser.add_argument('--constraints', type=str, default="", help='Percorso al file JSON con i vincoli utente')
    args = parser.parse_args()
    
    # Recupera l'input dell'utente
    user_input = args.activity
    print(f"Attività iniziale scelta dall'utente: {user_input}")
    
    # Carica i vincoli dal file JSON, se specificato
    user_constraints = []
    if args.constraints:
        try:
            with open(args.constraints, 'r') as f:
                user_constraints = json.load(f)
            print(f"Vincoli caricati: {user_constraints}")
        except Exception as e:
            print(f"Errore nel caricamento dei vincoli: {e}")

    # Configurazione del modello e del dispositivo
    model_name = "prajjwal1/bert-medium"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    dataset_path = "/kaggle/working/ProgettoTirocinio/dataset/BPIC15_1.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Errore: Il file CSV '{dataset_path}' non esiste!")

    # Campionamento del dataset per velocizzare l'esecuzione
    df = pd.read_csv(dataset_path, low_memory=False)
    print(f"Dataset ridotto: {len(df)} righe campionate.")

    model = BertClassifier(model_name, output_size=289).to(device)

    print("\nCaricamento del modello già addestrato...")
    model.load_state_dict(torch.load("/kaggle/working/modello_addestrato.pth", map_location=device))
    model.eval()

    # Valutazione del modello
    print("\nValutazione del modello sul test set...")
    dataset = load_dataset(dataset_path, tokenizer)
    dataset = dataset
    test_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    evaluate_model(model, test_loader, criterion, device)

    # Usa l'input dell'utente come attività iniziale
    initial_activities = [user_input]
    
    # Inizializza il filtro particellare con i vincoli dinamici
    pf = ParticleFilter(model, tokenizer, dataset.label_map, device, num_particles=3, constraints=user_constraints)
    pf.initialize_particles(initial_activities)
    final_particles = pf.run(steps=1)

    # Calcola la similarità CFld
    similarity_score = evaluate_log_similarity(model, tokenizer, dataset, dataset.label_map, device)
    print(f"CFld Similarity (dopo generazione tracce): {similarity_score:.4f}")

    # Stampa le particelle finali generate
    print("\nParticelle finali generate:")
    for particle in final_particles[:10]:  # Limita l'output a 10 particelle
        print([act.name for act in particle])
