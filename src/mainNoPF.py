import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import os
import time
import pm4py
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
from data_loader import load_dataset
from model import BertClassifier
from train import train
from evaluation import evaluate_model
from interactive_constraint_manager import InteractiveConstraintManager
from log_similarity import evaluate_log_similarity
from activity import ActivityPrediction
from predict import predict_next_log_with_constraints

from pm4py.objects.conversion.log import converter as log_converter
from Declare4Py.D4PyEventLog import D4PyEventLog
from declare.DeclareMiner import DeclareMiner
import logging
from constraints_checker import check_constraints
from constraints import constraints

# Disattiva i messaggi di DEBUG di Numba e CUDA
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)



def generate_trace(model, tokenizer, initial_activity, label_map, device, constraint_manager, max_steps=10):
    trace = [ActivityPrediction(initial_activity, 1.0)]
    print(f"Generazione traccia: inizio con '{initial_activity}'")

    for step in range(max_steps):
        candidates = predict_next_log_with_constraints(
            model, tokenizer, trace, label_map, device,
            num_candidates=5, constraint_manager=constraint_manager
        )

        if not candidates:
            print("Nessuna predizione valida, terminazione.")
            break

        next_act = max(candidates, key=lambda x: x.probability)
        trace.append(next_act)
        print(f"Step {step + 1}: aggiunto '{next_act.name}' con prob {next_act.probability:.4f}")

        if next_act.name == "END OF SEQUENCE":
            print("Traccia completata (END OF SEQUENCE raggiunto).")
            break

    return trace


def discover_constraints(csv_file, min_support, max_support, consider_vacuity=True, itemsets_support=0.9,
                         max_declare_cardinality=3):
    df = pd.read_csv(csv_file)
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
    model_name = "prajjwal1/bert-medium"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    dataset_path = "/kaggle/working/ProgettoTirocinio/dataset/helpdesk.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Errore: Il file CSV '{dataset_path}' non esiste!")

    dataset = load_dataset(dataset_path, tokenizer)
    model = BertClassifier(model_name, output_size=dataset.num_classes).to(device)

    model_path = "/kaggle/working/modello_addestrato-helpdesk.pth"
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

    print("\nValutazione del modello sul test set...")
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    evaluate_model(model, test_loader, criterion, device)

    discovered_constraints = discover_constraints(dataset_path, 0.7, 0.9)
    print("\nVincoli scoperti (supporto 70%-90%):")
    for constraint in discovered_constraints:
        print(constraint)

    constraint_manager = InteractiveConstraintManager()
    constraint_manager.user_constraints.extend(discovered_constraints)

    initial_activity = "Take in charge ticket"
    start_inference = time.time()

    generated_trace = generate_trace(
        model, tokenizer, initial_activity, dataset.label_map, device, constraint_manager, max_steps=10
    )

    end_inference = time.time()
    print(f"\n Tempo di inferenza (generazione traccia): {end_inference - start_inference:.2f} secondi.")

    print("\nTraccia generata:")
    print([act.name for act in generated_trace])

    similarity_score = evaluate_log_similarity([generated_trace], dataset.label_map, dataset.traces)
    print(f"\nCFld Similarity (dopo generazione tracce): {1 - similarity_score:.4f}")
    print(f"\nCFls Similarity (dopo generazione tracce): {similarity_score:.4f}")

