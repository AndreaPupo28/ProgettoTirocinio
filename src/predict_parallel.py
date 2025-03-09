import torch
import numpy as np
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction

def predict_parallel_sequences(model, tokenizer, initial_log, label_map, device, k=5): # consideriamo i 5 log più probabili
    model.eval()
    sequences = [[initial_log]]
    final_sequences = []

    while sequences:
        new_sequences = []
        #print(f"\nAttuali sequenze in elaborazione: {[' → '.join([act.name if isinstance(act, ActivityPrediction) else act for act in seq]) for seq in sequences]}")
        print(f"\nElaborazione di {len(sequences)} sequenze attive...")
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

            sorted_indices = np.argsort(probs)[::-1] # ordina gli indici delle probabilità in modo decrescente
            valid_candidates = []

            #print(f"\nPredizioni per la sequenza '{current_log}':")
            for idx in sorted_indices[:k]:
                candidate_log = list(label_map.keys())[idx]
                candidate_prob = probs[idx]
                new_sequence = current_log + " " + candidate_log
                #print(f"  Candidato: {candidate_log}, Probabilità: {candidate_prob:.4f}")

                if check_constraints(new_sequence, constraints, detailed=False, completed=True):
                    valid_candidates.append(ActivityPrediction(candidate_log, candidate_prob))

            if valid_candidates: # controlla se ci sono candidati che rispettano i vincoli
                for candidate in valid_candidates:
                    new_sequences.append(seq + [ActivityPrediction(candidate.name, candidate.probability)]) # concatena la sequenza corrente con la nuova attività
                    #print(f"  → Aggiunta nuova sequenza: {' → '.join([act.name if isinstance(act, ActivityPrediction) else act for act in seq + [candidate]])}")
            else:
                final_sequences.append(seq)
                print(f"  Nessun candidato valido, sequenza finale: {' → '.join(seq)}")

        sequences = new_sequences


    print("\nSequenze finali generate:")
    for seq in final_sequences:
        print(f"  {' → '.join(seq)}")

    return [[(activity.name, activity.probability) for activity in seq] for seq in final_sequences] #lista di tuple
