import torch
import numpy as np
from constraints_checker import check_constraints
from constraints import constraints


def predict_parallel_sequences(model, tokenizer, initial_log, label_map, device, k=3):
    model.eval()
    sequences = [[initial_log]]  # Lista di sequenze, inizialmente con una sola traccia
    final_sequences = []  # Lista che conterrà tutte le tracce generate

    while sequences:
        new_sequences = []
        print(f"\nAttuali sequenze in elaborazione: {[' → '.join(seq) for seq in sequences]}")
        for seq in sequences:
            current_log = " ".join(seq)
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

            sorted_indices = np.argsort(probs)[::-1]  # Ordina per probabilità decrescente
            valid_candidates = []

            print(f"\nPredizioni per la sequenza '{current_log}':")
            for idx in sorted_indices[:k]:  # Considera solo i top-k candidati
                candidate_log = list(label_map.keys())[idx]
                candidate_prob = probs[idx]
                new_sequence = current_log + " " + candidate_log
                print(f"  Candidato: {candidate_log}, Probabilità: {candidate_prob:.4f}")

                if check_constraints(new_sequence, constraints, detailed=False, completed=True):
                    valid_candidates.append((candidate_log, candidate_prob))

            if valid_candidates:
                for candidate, _ in valid_candidates:
                    new_sequences.append(seq + [candidate])
                    print(f"  → Aggiunta nuova sequenza: {' → '.join(seq + [candidate])}")
            else:
                final_sequences.append(seq)  # Se nessun candidato è valido, la sequenza è completa
                print(f"  Nessun candidato valido, sequenza finale: {' → '.join(seq)}")

        sequences = new_sequences  # Aggiorna la lista con le nuove sequenze

    print("\nSequenze finali generate:")
    for seq in final_sequences:
        print(f"  {' → '.join(seq)}")

    return final_sequences  # Ritorna tutte le sequenze generate
