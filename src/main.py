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
from ipywidgets import widgets
from IPython.display import display, clear_output
import json
import time
import numpy as np

if __name__ == "__main__":
    model_name = "prajjwal1/bert-medium"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    dataset_path = "/kaggle/working/ProgettoTirocinio/dataset/sepsis.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Errore: Il file CSV '{dataset_path}' non esiste!")

    df = pd.read_csv(dataset_path, low_memory=False)
    model = BertClassifier(model_name, output_size=len(set(df["activity"]))).to(device)

    # Widget per input dinamico dell'attività iniziale
    activity_widget = widgets.Text(
        value='validate request',
        placeholder='Inserisci un\'attività',
        description='Attività:',
        disabled=False
    )

    # Widget per aggiungere vincoli dinamici in formato JSON
    constraints_widget = widgets.Textarea(
        value='n',
        placeholder='Inserisci i vincoli in formato JSON o "n" per nessun vincolo',
        description='Vincoli:',
        disabled=False,
        layout=widgets.Layout(width='100%', height='100px')
    )

    save_button = widgets.Button(description="Salva Vincoli", button_style='success')
    output = widgets.Output()

    def on_save_button_clicked(b):
        with output:
            clear_output()
            constraints_data = constraints_widget.value.strip().lower()
            if constraints_data == "n":
                constraints = []
                with open('/kaggle/working/vincoli.json', 'w') as f:
                    json.dump(constraints, f)
                print("Nessun vincolo aggiuntivo selezionato. File 'vincoli.json' vuoto creato.")
            else:
                try:
                    constraints = json.loads(constraints_data)
                    with open('/kaggle/working/vincoli.json', 'w') as f:
                        json.dump(constraints, f)
                    print("Vincoli salvati correttamente in 'vincoli.json'.")
                except json.JSONDecodeError:
                    print("Errore: I vincoli non sono in un formato JSON valido.")
            print(f"Attività iniziale scelta: {activity_widget.value}")

    save_button.on_click(on_save_button_clicked)
    display(activity_widget, constraints_widget, save_button, output)

    initial_activity = activity_widget.value
    vincoli_path = '/kaggle/working/vincoli.json'
    if os.path.exists(vincoli_path):
        with open(vincoli_path, 'r') as f:
            constraints = json.load(f)
    else:
        constraints = []
    print(f"Attività iniziale scelta dall'utente: {initial_activity}")
    print(f"Vincoli caricati: {constraints}")

    if not os.path.exists("/kaggle/working/modello_addestrato-sepsis.pth"):
        print("\nAvvio dell'addestramento...")
        start_time = time.time()
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
        torch.save(model.state_dict(), "/kaggle/working/modello_addestrato-sepsis.pth")
        print("\nModello addestrato e salvato con successo.")
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Tempo totale di addestramento: {training_time:.2f} secondi")
    else:
        print("\nCaricamento del modello già addestrato...")
        model.load_state_dict(torch.load("/kaggle/working/modello_addestrato-sepsis.pth"))
        model.eval()

    print("\nValutazione del modello sul test set...")
    dataset = load_dataset(dataset_path, tokenizer)

    reduced_test_size = int(0.10 * len(dataset)) 
    reduced_indices = np.random.choice(len(dataset), reduced_test_size, replace=False)
    reduced_test_dataset = Subset(dataset, reduced_indices)

    test_loader = DataLoader(reduced_test_dataset, batch_size=8, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    evaluate_model(model, test_loader, criterion, device)

    initial_activities = [initial_activity]
    pf = ParticleFilter(model, tokenizer, dataset.label_map, device, num_particles=50)
    pf.initialize_particles(initial_activities)
    final_particles = pf.run(steps=2)

    similarity_score = evaluate_log_similarity(model, tokenizer, dataset, dataset.label_map, device)
    print(f"CFld Similarity (dopo generazione tracce): {similarity_score:.4f}")

    print("\nParticelle finali generate:")
    for particle in final_particles:
        print([act.name for act in particle])
