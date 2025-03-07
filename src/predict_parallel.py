import torch
import numpy as np
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction

import time
import logging
import numpy as np
import torch
from activity import ActivityPrediction
from constraints_checker import check_constraints
from constraints import constraints

def predict_parallel_sequences(model, tokenizer, initial_log, label_map, device, k=2, max_iterations=3):
    model.eval()
    sequences = [[initial_log]]
    final_sequences = []
    iteration = 0
    last_print_time = time.time()
    print_interval = 60  # stampa un messaggio ogni 60 secondi

    while sequences and iteration < max_iterations:
        iteration += 1
        new_sequences = []
        
        # Log delle prime 3 sequenze
        sequences_to_log = []
        for i, seq in enumerate(sequences[:3]):
            seq_str = " → ".join([act.name if isinstance(act, ActivityPrediction) else act for act in seq])
            sequences_to_log.append(f"{i+1}) {seq_str}")
        logging.info("Attuali sequenze in elaborazione (prime 3): " + "; ".join(sequences_to_log))
        if len(sequences) > 3:
            logging.info(f"... e altre {len(sequences)-3} sequenze non visualizzate.")

        # Stampa periodica sullo stdout
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            print(f"Progresso: Iterazione {iteration} in corso, {len(sequences)} sequenze attive.")
            last_print_time = current_time

        for seq in sequences:
            current_log = " → ".join([act.name if isinstance(act, ActivityPrediction) else act for act in seq])
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
            prob_threshold = 0.2  # Considera solo candidati con probabilità >= 0.2
            for idx in sorted_indices[:k]:
                candidate_log = list(label_map.keys())[idx]
                candidate_prob = probs[idx]
                if candidate_prob < prob_threshold:
                    continue
                new_sequence = current_log + " " + candidate_log
                if check_constraints(new_sequence, constraints, detailed=False, completed=True):
                    valid_candidates.append(ActivityPrediction(candidate_log, candidate_prob))
                    
            if valid_candidates:
                for candidate in valid_candidates:
                    new_sequences.append(seq + [ActivityPrediction(candidate.name, candidate.probability)])
            else:
                final_sequences.append(seq)
                logging.info(f"Nessun candidato valido, sequenza finale: {' → '.join([a.name if isinstance(a, ActivityPrediction) else a for a in seq])}")

        sequences = new_sequences

    logging.info("Sequenze finali generate:")
    for seq in final_sequences:
        print("  " + " → ".join([a.name if isinstance(a, ActivityPrediction) else a for a in seq]))

    return [[(activity.name, activity.probability) for activity in seq] for seq in final_sequences]

