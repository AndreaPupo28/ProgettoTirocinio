import torch
import numpy as np
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction

def predict_parallel_sequences(model, tokenizer, initial_log, label_map, device, k=2, max_iterations=3):
    model.eval()
    sequences = [[initial_log]]
    final_sequences = []
    iteration = 0

    while sequences and iteration < max_iterations:
        iteration += 1
        new_sequences = []
        
        # --- Stampa solo le prime 3 sequenze ---
        sequences_to_print = sequences[:3]
        print("\nAttuali sequenze in elaborazione (prime 3):")
        for i, seq in enumerate(sequences_to_print):
            seq_str = " → ".join(
                [act.name if isinstance(act, ActivityPrediction) else act for act in seq]
            )
            print(f"  {i+1}) {seq_str}")
        if len(sequences) > 3:
            print(f"... e altre {len(sequences) - 3} sequenze non visualizzate.")
        # --- Fine stampa ---
        
        for seq in sequences:
            current_log = " → ".join(
                [act.name if isinstance(act, ActivityPrediction) else act for act in seq]
            )
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

            sorted_indices = np.argsort(probs)[::-1]  # ordina gli indici delle probabilità in modo decrescente
            valid_candidates = []
            prob_threshold = 0.2  # Considera solo candidati con probabilità >= 0.2
            for idx in sorted_indices[:k]:
                candidate_log = list(label_map.keys())[idx]
                candidate_prob = probs[idx]
                if candidate_prob < prob_threshold:
                    continue  # Scarta candidati con probabilità troppo bassa
                new_sequence = current_log + " " + candidate_log
                if check_constraints(new_sequence, constraints, detailed=False, completed=True):
                    valid_candidates.append(ActivityPrediction(candidate_log, candidate_prob))
                    
            if valid_candidates:
                for candidate in valid_candidates:
                    new_sequences.append(seq + [ActivityPrediction(candidate.name, candidate.probability)])
            else:
                final_sequences.append(seq)
                print(f"  Nessun candidato valido, sequenza finale: {' → '.join(seq)}")

        sequences = new_sequences

    print("\nSequenze finali generate:")
    for seq in final_sequences:
        print(f"  {' → '.join(seq)}")

    return [[(activity.name, activity.probability) for activity in seq] for seq in final_sequences]


