import numpy as np
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from scipy.optimize import linear_sum_assignment

def generate_log_matrix(logs, label_map):
    """
    Converte una lista di tracce (liste di stringhe) in una matrice binaria.
    Ogni riga rappresenta una traccia e ogni colonna un'attività.
    """
    matrix = np.zeros((len(logs), len(label_map)), dtype=int)
    for i, trace in enumerate(logs):
        for activity in trace:
            if activity in label_map:
                matrix[i, label_map[activity]] = 1
    return matrix

def evaluate_log_similarity(final_particles, label_map):
    """
    Calcola la similarità (CFld) tra le tracce generate dalla Particle Filter.
    Il parametro final_particles è una lista di sequenze generate (ogni sequenza è una lista
    di oggetti ActivityPrediction). Questa funzione non genera ulteriori tracce.
    """
    # Estrai i nomi delle attività da ogni particella
    generated_traces = [[activity.name for activity in particle] for particle in final_particles]
    # Crea la matrice log del log generato
    generated_log_matrix = generate_log_matrix(generated_traces, label_map)
    # Per il confronto, qui confrontiamo il log generato con se stesso, ottenendo una similarità pari a 1.
    # Se desideri confrontare con un log originale, dovrai passare anche quel log.
    similarity = get_log_similarity(generated_log_matrix, generated_log_matrix)
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
