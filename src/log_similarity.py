import numpy as np
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from scipy.optimize import linear_sum_assignment
from predict import predict_next_log_with_constraints

def generate_log_matrix(logs, label_map):
    """
    Genera una matrice numpy dove ogni riga rappresenta una traccia
    e ogni colonna rappresenta una attività nel label_map.
    """
    matrix = np.zeros((len(logs), len(label_map)), dtype=int)
    for i, trace in enumerate(logs):
        for activity in trace:
            if activity in label_map:
                matrix[i, label_map[activity]] = 1
    return matrix


def evaluate_log_similarity(model, tokenizer, dataset, label_map, device, num_candidates=2):
    """
    Valuta la similarità tra il log originale e quello generato usando la metrica cfld.
    """
    # Matrice del log originale
    original_traces = [trace[0] for trace in dataset.data]
    original_log_matrix = generate_log_matrix(original_traces, label_map)

    # Generazione del log predetto
    generated_traces = []
    for trace in original_traces:
        input_text = " ".join(trace)
        predicted_sequences = predict_next_log_with_constraints(
            model, tokenizer, input_text, label_map, device, num_candidates
        )
        if predicted_sequences and predicted_sequences[0]:
            generated_traces.append([pred[0] for pred in predicted_sequences[0]])

    # Matrice del log generato
    generated_log_matrix = generate_log_matrix(generated_traces, label_map)

    # Calcolo della similarità tra i log
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
    total_distance = 0
    for i, j in zip(row_ind, col_ind):
        total_distance += cost_matrix[i][j]

    cfld = total_distance / len(row_ind)
    return cfld


def _pair_traces(normalized_distances, original_log, generated_log):
    cost_matrix = np.array(normalized_distances).reshape(len(original_log), len(generated_log))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix, row_ind, col_ind

def get_log_similarity(original_log, generated_log):
    normalized_distances = _compute_pair_distances(original_log, generated_log)
    cost_matrix, row_ind, col_ind = _pair_traces(normalized_distances, original_log, generated_log)
    cfld_metric = _compute_cfld(row_ind, col_ind, cost_matrix)
    return 1 - cfld_metric
