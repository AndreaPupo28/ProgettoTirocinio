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
    
    # Percorso del dataset
    dataset_path = "/kaggle/working/ProgettoTirocinio/dataset/helpdesk.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Errore: Il file CSV '{dataset_path}' non esiste!")
    
    # Caricamento del dataset per determinare le dimensioni dell'output
    df = pd.read_csv(dataset_path, low_memory=False)
    
    # Istanzia il modello con il numero di classi pari al numero di attività uniche nel dataset
    model = BertClassifier(model_name, output_size=len(set(df["activity"]))).to(device)
    
    # Controllo sull'esistenza del file del modello addestrato
    model_path = "/kaggle/working/modello_addestrato2.pth"
    if not os.path.exists(model_path):
        print("\nAvvio dell'addestramento...")
        # Carica il dataset per l'addestramento e la valutazione
        dataset = load_dataset(dataset_path, tokenizer)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
    
        model = train(model, train_loader, optimizer, 10, criterion, device)
        # Assicura la creazione della cartella di destinazione, se necessario
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print("\nModello addestrato e salvato con successo.")
    else:
        print("\nCaricamento del modello già addestrato...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    
    # Valutazione del modello sul test set
    print("\nValutazione del modello sul test set...")
    dataset = load_dataset(dataset_path, tokenizer)
    test_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    evaluate_model(model, test_loader, criterion, device)
    
    # Usa l'input dell'utente come attività iniziale
    initial_activities = [user_input]
    
    # Inizializza il filtro particellare con i vincoli dinamici (se presenti)
    pf = ParticleFilter(model, tokenizer, dataset.label_map, device, num_particles=3, constraints=user_constraints)
    pf.initialize_particles(initial_activities)
    final_particles = pf.run(steps=1)
    
    # Calcola la similarità CFld
    similarity_score = evaluate_log_similarity(model, tokenizer, dataset, dataset.label_map, device)
    print(f"CFld Similarity (dopo generazione tracce): {similarity_score:.4f}")
    
    # Salva tutte le particelle in un file di testo (opzionale, utile se vuoi l'intero output)
    with open("output_particles.txt", "w") as f:
        for particle in final_particles:
            f.write(str([act.name for act in particle]) + "\n")

    # Stampa limitata delle particelle finali generate (es. prime 10) per evitare log troppo lunghi
    print("\nParticelle finali generate (prime 10):")
    for i, particle in enumerate(final_particles):
        if i < 4:
            print([act.name for act in particle])
        else:
            break
    
    # Messaggio di debug finale
    print("\nFine dello script. Tutto completato con successo!")
