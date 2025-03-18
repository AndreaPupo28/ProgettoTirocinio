# import torch
# import os
# import pandas as pd
# from transformers import AutoTokenizer
# from torch.utils.data import DataLoader
# from data_loader import load_dataset
# from model import BertClassifier
# from train import train
# from evaluation import evaluate_model
# from particle_filter import ParticleFilter
# from log_similarity import evaluate_log_similarity
# from ipywidgets import widgets
# from IPython.display import display, clear_output
# import json
# import time
# import numpy as np
# from torch.utils.data import Subset
#
#
# if __name__ == "__main__":
#     model_name = "prajjwal1/bert-medium"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
#     dataset_path = "/kaggle/working/ProgettoTirocinio/dataset/BPIC15_1.csv"
#
#     if not os.path.exists(dataset_path):
#         raise FileNotFoundError(f"Errore: Il file CSV '{dataset_path}' non esiste!")
#
#     df = pd.read_csv(dataset_path, low_memory=False)
#     model = BertClassifier(model_name, output_size=len(set(df["activity"]))).to(device)
#
#     # Widget per input dinamico dell'attività iniziale
#     activity_widget = widgets.Text(
#         value='send letter in progress',
#         placeholder='Inserisci un\'attività',
#         description='Attività:',
#         disabled=False
#     )
#
#     # Widget per aggiungere vincoli dinamici in formato JSON
#     constraints_widget = widgets.Textarea(
#         value='n',
#         placeholder='Inserisci i vincoli in formato JSON o "n" per nessun vincolo',
#         description='Vincoli:',
#         disabled=False,
#         layout=widgets.Layout(width='100%', height='100px')
#     )
#
#     save_button = widgets.Button(description="Salva Vincoli", button_style='success')
#     output = widgets.Output()
#
#     def on_save_button_clicked(b):
#         with output:
#             clear_output()
#             constraints_data = constraints_widget.value.strip().lower()
#             if constraints_data == "n":
#                 constraints = []
#                 with open('/kaggle/working/vincoli.json', 'w') as f:
#                     json.dump(constraints, f)
#                 print("Nessun vincolo aggiuntivo selezionato. File 'vincoli.json' vuoto creato.")
#             else:
#                 try:
#                     constraints = json.loads(constraints_data)
#                     with open('/kaggle/working/vincoli.json', 'w') as f:
#                         json.dump(constraints, f)
#                     print("Vincoli salvati correttamente in 'vincoli.json'.")
#                 except json.JSONDecodeError:
#                     print("Errore: I vincoli non sono in un formato JSON valido.")
#             print(f"Attività iniziale scelta: {activity_widget.value}")
#
#     save_button.on_click(on_save_button_clicked)
#     display(activity_widget, constraints_widget, save_button, output)
#
#     initial_activity = activity_widget.value
#     vincoli_path = '/kaggle/working/vincoli.json'
#     if os.path.exists(vincoli_path):
#         with open(vincoli_path, 'r') as f:
#             constraints = json.load(f)
#     else:
#         constraints = []
#     print(f"Attività iniziale scelta dall'utente: {initial_activity}")
#     print(f"Vincoli caricati: {constraints}")
#
#     if not os.path.exists("/kaggle/working/modello_addestrato.pth"):
#         print("\nAvvio dell'addestramento...")
#         start_time = time.time()
#         dataset = load_dataset(dataset_path, tokenizer)
#         train_size = int(0.8 * len(dataset))
#         test_size = len(dataset) - train_size
#         train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#
#         train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
#
#         optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
#         criterion = torch.nn.CrossEntropyLoss()
#
#         model = train(model, train_loader, optimizer, 10, criterion, device)
#         os.makedirs("models", exist_ok=True)
#         torch.save(model.state_dict(), "/kaggle/working/modello_addestrato.pth")
#         print("\nModello addestrato e salvato con successo.")
#         end_time = time.time()
#         training_time = end_time - start_time
#         print(f"Tempo totale di addestramento: {training_time:.2f} secondi")
#     else:
#         print("\nCaricamento del modello già addestrato...")
#         model.load_state_dict(torch.load("/kaggle/working/modello_addestrato.pth"))
#         model.eval()
#
    # print("\nValutazione del modello sul test set...")
    # dataset = load_dataset(dataset_path, tokenizer)
    #
    # reduced_test_size = int(0.50 * len(dataset))
    # reduced_indices = np.random.choice(len(dataset), reduced_test_size, replace=False)
    # reduced_test_dataset = Subset(dataset, reduced_indices)
    #
    # test_loader = DataLoader(reduced_test_dataset, batch_size=8, shuffle=False)
    #
    # criterion = torch.nn.CrossEntropyLoss()
    # evaluate_model(model, test_loader, criterion, device)
    #
    # pf = ParticleFilter(model, tokenizer, dataset.label_map, device, k=3)
    # pf.initialize_particles(initial_activity)
    # final_particles = pf.run(steps=4)
    #
    # similarity_score = evaluate_log_similarity(final_particles, dataset.label_map)
    # print(f"CFld Similarity (dopo generazione tracce): {similarity_score:.4f}")
    #
    # print("\nParticelle finali generate:")
    # for particle in final_particles:
    #     print([act.name for act in particle])

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import os
import time
import numpy as np
import pm4py
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
from data_loader import load_dataset
from model import BertClassifier
from train import train
from evaluation import evaluate_model
from predict_parallel import predict_parallel_sequences  # Funzione per predire la prossima attività con vincoli
from interactive_constraint_manager import InteractiveConstraintManager
from log_similarity import evaluate_log_similarity
from activity import ActivityPrediction
from predict import predict_next_log_with_constraints

# Import necessari per il discovery
from pm4py.objects.conversion.log import converter as log_converter
from Declare4Py.D4PyEventLog import D4PyEventLog
from declare.DeclareMiner import DeclareMiner
import logging

# Disattiva i messaggi di DEBUG di Numba e CUDA
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)



def generate_trace(model, tokenizer, initial_activity, label_map, device, constraint_manager, max_steps=10):
    """
    Genera una traccia (sequenza di ActivityPrediction) a partire dall'attività iniziale.
    Ad ogni step viene utilizzata la funzione predict_parallel_sequences per ottenere candidati
    che rispettano i vincoli; viene selezionato il candidato con probabilità maggiore.
    La generazione si interrompe se viene predetto "END OF SEQUENCE" o se si raggiunge max_steps.
    """
    trace = [ActivityPrediction(initial_activity, 1.0)]
    print(f"Generazione traccia: inizio con '{initial_activity}'")

    for step in range(max_steps):
        # Usa la funzione di predizione per ottenere i candidati (qui prendiamo k=3)
        candidates = predict_next_log_with_constraints(
            model, tokenizer, trace, label_map, device,
            num_candidates=5, constraint_manager=constraint_manager
        )

        if not candidates:
            print("Nessuna predizione valida, terminazione.")
            break

        # Seleziona il candidato con la probabilità più alta
        next_act = max(candidates, key=lambda x: x.probability)
        trace.append(next_act)
        print(f"Step {step + 1}: aggiunto '{next_act.name}' con prob {next_act.probability:.4f}")

        if next_act.name == "END OF SEQUENCE":
            print("Traccia completata (END OF SEQUENCE raggiunto).")
            break

    return trace


def discover_constraints(csv_file, min_support, max_support, consider_vacuity=True, itemsets_support=0.9,
                         max_declare_cardinality=3):
    """
    Data la path di un CSV contenente il log, converte il DataFrame in un event log (D4PyEventLog)
    e utilizza il DeclareMiner per scoprire i vincoli che sono soddisfatti in una percentuale di tracce compresa
    tra min_support e max_support.

    Se le colonne richieste ("Case ID", "Activity", "Start Timestamp") non sono presenti, si assume che il CSV
    contenga invece "case", "activity" e "timestamp" e si effettua una rinominazione.

    Ritorna la lista dei vincoli (constraints) scoperti.
    """
    df = pd.read_csv(csv_file)
    # Se non sono presenti le colonne attese, proviamo a rinominarle
    if not {"Case ID", "Activity", "Start Timestamp"}.issubset(df.columns):
        df = df.rename(columns={"case": "Case ID", "activity": "Activity", "timestamp": "Start Timestamp"})
    df = pm4py.format_dataframe(df, case_id="Case ID", activity_key="Activity", timestamp_key="Start Timestamp")

    event_log = D4PyEventLog(case_name="case:concept:name")
    event_log.log = pm4py.convert_to_event_log(log_converter.apply(df))
    event_log.log_length = len(event_log.log)
    event_log.timestamp_key = event_log.log._properties["pm4py:param:timestamp_key"]
    event_log.activity_key = event_log.log._properties["pm4py:param:activity_key"]

    discovery = DeclareMiner(
        log=event_log,
        consider_vacuity=consider_vacuity,
        min_support=min_support,
        max_support=max_support,
        itemsets_support=itemsets_support,
        max_declare_cardinality=max_declare_cardinality
    )
    discovered_model = discovery.run()
    return discovered_model.constraints


if __name__ == "__main__":
    # Parametri di setup
    model_name = "prajjwal1/bert-medium"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    dataset_path = "/kaggle/working/ProgettoTirocinio/dataset/sepsis.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Errore: Il file CSV '{dataset_path}' non esiste!")

    # Carica il dataset e crea il modello in base al numero di classi
    dataset = load_dataset(dataset_path, tokenizer)
    model = BertClassifier(model_name, output_size=dataset.num_classes).to(device)

    # Addestramento del modello
    model_path = "/kaggle/working/modello_addestrato-sepsis.pth"
    if not os.path.exists(model_path):
        print("\nAvvio dell'addestramento...")
        start_time = time.time()
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        model = train(model, train_loader, optimizer, epochs=10, criterion=criterion, device=device)
        torch.save(model.state_dict(), model_path)
        end_time = time.time()
        print(f"Modello addestrato in {end_time - start_time:.2f} secondi.")
    else:
        print("\nCaricamento del modello già addestrato...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    # Valutazione del modello sul test set
    print("\nValutazione del modello sul test set...")
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    evaluate_model(model, test_loader, criterion, device)

    # DISCOVERY: utilizza il nuovo codice per scoprire i vincoli
    discovered_constraints = discover_constraints(dataset_path, 0.7, 0.9)
    print("\nVincoli scoperti (supporto 70%-90%):")
    for constraint in discovered_constraints:
        print(constraint)

    # Integra i vincoli scoperti nel constraint manager (senza alterare il resto del sistema)
    constraint_manager = InteractiveConstraintManager()
    constraint_manager.user_constraints.extend(discovered_constraints)

    # Generazione della traccia con l'LM (senza Particle Filter)
    initial_activity = "CRP"  # Imposta l'attività iniziale desiderata
    generated_trace = generate_trace(model, tokenizer, initial_activity, dataset.label_map, device, constraint_manager,
                                     max_steps=10)

    print("\nTraccia generata:")
    print([act.name for act in generated_trace])

    # Calcolo della similarità CFld: confronta la traccia generata con le tracce originali del dataset
    similarity_score = evaluate_log_similarity([generated_trace], dataset.label_map, dataset.traces)
    print(f"\nCFld Similarity: {similarity_score:.4f}")
