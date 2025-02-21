import torch
import numpy as np
# Importa la funzione check_constraints dal modulo contraints_checker
from contraints_checker import check_constraints
# Importa la struttura dei constraints
from constraints import constraints


def predict_next_log_with_constraints(model, tokenizer, current_log, label_map, device, num_candidates=10):
    """
    Predice il prossimo log, controllando fra i top `num_candidates` (default 10)
    quale, se aggiunto alla sequenza corrente, rispetti i constraints definiti.
    """
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            current_log,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        ).to(device)
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy().flatten()

    # Ordina gli indici in ordine decrescente di probabilità
    sorted_indices = np.argsort(probs)[::-1]
    valid_candidates = []  # Lista dei candidati validi: tuple (nome log, probabilità)

    for idx in sorted_indices[:num_candidates]:
        candidate_log = list(label_map.keys())[idx]
        candidate_prob = probs[idx]
        # Crea la nuova sequenza concatenando quella corrente e il candidato
        new_sequence = current_log + " " + candidate_log

        # Verifica se la nuova sequenza rispetta i constraints definiti
        if check_constraints(new_sequence, constraints, detailed=False, completed=True):
            valid_candidates.append((candidate_log, candidate_prob))

    if valid_candidates:
        # Se ci sono candidati validi, ne viene scelto il primo (quello con la probabilità più alta)
        chosen_candidate = valid_candidates[0][0]
        print("Candidati validi che soddisfano i constraints:", valid_candidates)
        return chosen_candidate, probs
    else:
        print(f"Nessun candidato tra i top {num_candidates} soddisfa i constraints.")
        return None, probs
