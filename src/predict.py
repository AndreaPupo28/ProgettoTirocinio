import torch
import numpy as np
from contraints_checker import check_constraints
from constraints import constraints


def predict_next_log_with_constraints(model, tokenizer, current_log, label_map, device, num_candidates=10):
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

    sorted_indices = np.argsort(probs)[::-1]
    valid_candidates = []

    for idx in sorted_indices[:num_candidates]:
        candidate_log = list(label_map.keys())[idx]
        candidate_prob = probs[idx]
        new_sequence = current_log + " " + candidate_log

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
