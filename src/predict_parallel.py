import torch
import numpy as np
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction


def predict_parallel_sequences(model, tokenizer, current_sequence, label_map, device, k):
    """
    Genera le prossime attività più probabili per una sequenza corrente.
    Questa funzione ora esegue SOLO UNA previsione per step, senza generare subito tutte le sequenze.
    """
    model.eval()

    input_text = " ".join([act.name for act in current_sequence])
    with torch.no_grad():
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        ).to(device)
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy().flatten()

    sorted_indices = np.argsort(probs)[::-1]  # Ordina le probabilità in ordine decrescente
    valid_candidates = []

    print(f"\n[INFO] Predizioni per la sequenza '{input_text}':")

    for idx in sorted_indices[:k]:  # Seleziona solo i top-k candidati
        candidate_log = list(label_map.keys())[idx]
        candidate_prob = probs[idx]
        new_sequence = input_text + " " + candidate_log
        print(f"  - Candidato: {candidate_log}, Probabilità: {candidate_prob:.4f}")

        if check_constraints(new_sequence, constraints, detailed=False, completed=True):
            valid_candidates.append(ActivityPrediction(candidate_log, candidate_prob))

    return valid_candidates  # Ritorna solo le nuove attività possibili
