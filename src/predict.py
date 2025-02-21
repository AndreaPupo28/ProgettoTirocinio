import torch
import numpy as np
from collections import deque
from constraints_checker import check_constraints
from constraints import constraints


def generate_traces(model, tokenizer, initial_log, label_map, device, num_candidates=10):
    """
    Genera più tracce contemporaneamente rispettando i vincoli.
    - Se ci sono più attività valide tra le prime 10 più probabili, si generano più tracce.
    - Ogni traccia viene espansa iterativamente con nuove attività valide.
    """

    model.eval()

    # Coda per gestire le tracce (FIFO)
    trace_queue = deque()
    trace_queue.append([initial_log])  # Inseriamo la prima attività come sequenza iniziale

    final_traces = []  # Lista per salvare le tracce complete

    while trace_queue:
        current_trace = trace_queue.popleft()  # Estraiamo la traccia più vecchia
        current_log = " ".join(current_trace)  # Convertiamo la lista in stringa

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

        # Otteniamo le attività più probabili
        sorted_indices = np.argsort(probs)[::-1]
        valid_candidates = []

        # Selezioniamo le attività che rispettano i vincoli
        for idx in sorted_indices[:num_candidates]:
            candidate_activity = list(label_map.keys())[idx]
            candidate_prob = probs[idx]
            new_trace = current_trace + [candidate_activity]  # Creiamo la nuova traccia

            # Verifichiamo se la nuova traccia rispetta i vincoli
            if check_constraints(" ".join(new_trace), constraints, detailed=False, completed=True):
                valid_candidates.append((new_trace, candidate_prob))

        if valid_candidates:
            for new_trace, _ in valid_candidates:
                trace_queue.append(new_trace)  # Aggiungiamo la nuova traccia alla coda per l'espansione
        else:
            # Nessuna nuova attività valida → la traccia è completa
            final_traces.append(current_trace)

    return final_traces


def predict_next_log_with_constraints(model, tokenizer, current_log, label_map, device, num_candidates=10):
    """
    Mantiene la vecchia funzionalità di predizione di un singolo log con vincoli.
    - Se il vecchio codice prevedeva una singola attività, questa funzione continua a supportarlo.
    - Questa funzione può essere usata quando serve una sola predizione invece della generazione di tracce multiple.
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
    valid_candidates = []

    for idx in sorted_indices[:num_candidates]:
        candidate_log = list(label_map.keys())[idx]
        candidate_prob = probs[idx]

        # Crea la nuova sequenza aggiungendo il candidato
        new_sequence = current_log + " " + candidate_log

        # Verifica se la nuova sequenza rispetta i vincoli
        if check_constraints(new_sequence, constraints, detailed=False, completed=True):
            valid_candidates.append((candidate_log, candidate_prob))

    if valid_candidates:
        # Scegliamo il candidato con probabilità massima tra quelli validi
        chosen_candidate = valid_candidates[0][0]
        return chosen_candidate, probs
    else:
        return None, probs
