import torch
import numpy as np
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction

def predict_parallel_sequences(model, tokenizer, current_sequence, label_map, device, k):
    """
    Genera le prossime attività più probabili per una sequenza corrente.
    Assicura che la sequenza sia sempre composta da oggetti ActivityPrediction.
    """
    model.eval()

    # Se `current_sequence` è una stringa, la convertiamo in una lista di un solo elemento ActivityPrediction
    if isinstance(current_sequence, str):
        current_sequence = [ActivityPrediction(current_sequence, 1.0)]
    elif isinstance(current_sequence, list) and all(isinstance(act, str) for act in current_sequence):
        current_sequence = [ActivityPrediction(act, 1.0) for act in current_sequence]

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

    valid_candidates = []
    for idx in sorted_indices:
        if len(valid_candidates) >= k:
            break
        # Seleziona solo i top-k candidati
        candidate_log = list(label_map.keys())[idx]
        candidate_prob = probs[idx]
        new_sequence = input_text + " " + candidate_log
        print(f"  - Candidato: {candidate_log}, Probabilità: {candidate_prob:.4f}")

        if check_constraints(new_sequence, constraints, detailed=False, completed=True):
            valid_candidates.append(ActivityPrediction(candidate_log, candidate_prob))

    return valid_candidates  # Ritorna solo le nuove attività possibili
