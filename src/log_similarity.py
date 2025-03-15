import numpy as np
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from scipy.optimize import linear_sum_assignment
from predict import predict_next_log_with_constraints


def generate_log_matrix(logs, label_map):
    """
    Converte una lista di tracce (liste di attività) in una matrice binaria.
    Ogni riga rappresenta una traccia e ogni colonna un'attività.
    """
    matrix = np.zeros((len(logs), len(label_map)), dtype=int)
    for i, trace in enumerate(logs):
        for activity in trace:
            if activity in label_map:
                matrix[i, label_map[activity]] = 1
    return matrix


def evaluate_log_similarity(model, tokenizer, dataset, label_map, device, num_candidates=2, max_traces=100):
    """
    Valuta la similarità tra il log originale e quello predetto.
    Per evitare la generazione infinita, viene utilizzato solo un campione (max_traces) di tracce dal dataset.
    """
    # Prendi un campione delle tracce originali (gli input, ossia le sequenze senza l'attività successiva)
    original_traces_all = [trace[0] for trace in dataset.data]
    original_traces = original_traces_all[:max_traces]
    original_log_matrix = generate_log_matrix(original_traces, label_map)

    # Generazione del log predetto: per ogni traccia del campione, genera la previsione one-shot
    generated_traces = []
    for trace in original_traces:
        input_text = " ".join(trace)
        predicted_sequences = predict_next_log_with_constraints(
            model, tokenizer, input_text, label_map, device, num_candidates
        )
        if predicted_sequences and predicted_sequences[0]:
            if isinstance(predicted_sequences[0], list):
                generated_traces.append([pred.name for pred in predicted_sequences[0]])
            else:
                generated_traces.append([predicted_sequences[0].name])

    generated_log_matrix = generate_log_matrix(generated_traces, label_map)
    similarity = get_log_similarity(original_log_matrix, generated_log_matrix)
    return similarity


def _compute_pair_distances(original_log, generated_log):
    distances = []
    for i, original_trace in enumerate(original_log):
        for j, generated_trace in enumerate(generated_log):
            distance = normalized_damerau_levenshtein_distance(original_trace, generated_trace)
            distances.append(distance)
    return distances


def _compute_cfld(row_ind, col_ind, cost_matrix):
    if len(row_ind) == 0:
        return 0
    total_distance = sum(cost_matrix[i][j] for i, j in zip(row_ind, col_ind))
    return total_distance / len(row_ind)


def _pair_traces(normalized_distances, original_log, generated_log):
    cost_matrix = np.array(normalized_distances).reshape(len(original_log), len(generated_log))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix, row_ind, col_ind


def get_log_similarity(original_log, generated_log):
    normalized_distances = _compute_pair_distances(original_log, generated_log)
    cost_matrix, row_ind, col_ind = _pair_traces(normalized_distances, original_log, generated_log)
    cfld_metric = _compute_cfld(row_ind, col_ind, cost_matrix)
    return 1 - cfld_metric
