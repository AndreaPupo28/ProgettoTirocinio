import torch
import numpy as np
from constraints_checker import satisfies


def predict_next_log(model, tokenizer, current_log, label_map, device, num_particles=100, completed=True):
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

    sorted_indices = np.argsort(probs)[::-1]  # Indici delle classi ordinate per probabilità decrescente
    valid_activities = []  # Per memorizzare solo le attività che rispettano i vincoli

    for idx in sorted_indices:
        log_name = list(label_map.keys())[idx]
        log_prob = probs[idx]

        # Creiamo una sequenza estesa con la nuova attività per verificare i vincoli
        extended_sequence = f"{current_log} {log_name}"
        
        dummy_constraint = {
            "template": {"is_binary": False, "supports_cardinality": False},  # Evitiamo errori sulle proprietà mancanti
            "activities": [],
            "condition": ["dummy_condition"]
        }

        if satisfies(extended_sequence, dummy_constraint, detailed=False, completed=completed):
            valid_activities.append((log_name, log_prob))

    if not valid_activities:
        print("Nessuna attività predetta soddisfa i vincoli. Terminazione della generazione della sequenza.")
        return None, None  # Se nessuna attività è valida, interrompiamo la generazione

    # Ricalcoliamo la probabilità solo per le attività valide
    valid_probs = np.array([prob for _, prob in valid_activities])
    valid_probs /= valid_probs.sum()  # Normalizziamo le probabilità

    valid_labels = [label for label, _ in valid_activities]

    # Monte Carlo sampling sulle attività valide
    particles = np.random.choice(valid_labels, size=num_particles, p=valid_probs)
    unique, counts = np.unique(particles, return_counts=True)
    most_likely = unique[np.argmax(counts)]  # L'attività più scelta

    return most_likely, valid_probs
