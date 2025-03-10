import torch
import numpy as np
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction

process_terminated = False

def predict_parallel_sequences(model, tokenizer, initial_log, label_map, device, k=5):
    global process_terminated
    model.eval()
    sequences = [[initial_log]]
    final_sequences = []
    
    MAX_SEQUENCES = 40000
    prev_len_sequences = 0

    while sequences and len(final_sequences) < MAX_SEQUENCES and not process_terminated:
        if len(sequences) > MAX_SEQUENCES:
            print(f"Limite massimo di {MAX_SEQUENCES} sequenze attive raggiunto. Troncamento in corso...")
            sequences = sequences[:MAX_SEQUENCES]

        new_sequences = []
        print(f"\nElaborazione di {len(sequences)} sequenze attive...")

        for seq in sequences:
            if len(new_sequences) >= MAX_SEQUENCES:
                print("Raggiunto il limite di nuove sequenze. Interruzione dell'aggiunta di nuove particelle.")
                break

            current_log = "<SOS> " + " → ".join([act.name if isinstance(act, ActivityPrediction) else act for act in seq]) + " <EOS>"
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

            for idx in sorted_indices[:k]:
                candidate_log = list(label_map.keys())[idx]
                candidate_prob = probs[idx]
                new_sequence = current_log + " " + candidate_log

                if check_constraints(new_sequence, constraints, detailed=False, completed=True):
                    valid_candidates.append(ActivityPrediction(candidate_log, candidate_prob))

            if valid_candidates:
                for candidate in valid_candidates:
                    if len(new_sequences) < MAX_SEQUENCES:
                        new_sequences.append(seq + [ActivityPrediction(candidate.name, candidate.probability)])
                    else:
                        print("Raggiunto il limite durante l'aggiunta di nuovi candidati. Interruzione.")
                        break
            else:
                final_sequences.append(seq)
                print(f"  Nessun candidato valido, sequenza finale: {' → '.join([act.name if isinstance(act, ActivityPrediction) else act for act in seq])}")

        if len(new_sequences) == prev_len_sequences:
            print("Nessuna nuova sequenza valida generata. Interruzione del ciclo.")
            process_terminated = True
            break

        prev_len_sequences = len(new_sequences)
        sequences = new_sequences

    return [[(activity.name, activity.probability) for activity in seq] for seq in final_sequences]

