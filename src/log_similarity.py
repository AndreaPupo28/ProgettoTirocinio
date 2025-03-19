import numpy as np
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from joblib import Parallel, delayed

def _compute_distance_for_original(original_trace, generated_log):
    return [normalized_damerau_levenshtein_distance(original_trace, gen_trace)
            for gen_trace in generated_log]

def _compute_pair_distances_parallel(original_log, generated_log, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_distance_for_original)(orig, generated_log)
        for orig in original_log
    )
    # Appiattisce la lista: prima tutte le distanze per la prima traccia, poi per la seconda, etc.
    distances = [d for sublist in results for d in sublist]
    return distances


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


def evaluate_log_similarity(final_particles, label_map, original_traces):
    """
    Calcola la similarità (CFld) confrontando le tracce generate dalla Particle Filter
    con le tracce originali del dataset.

    Parametri:
    - final_particles: lista di sequenze generate (ogni sequenza è una lista di oggetti ActivityPrediction).
    - label_map: dizionario che mappa le attività agli indici.
    - original_traces: lista delle tracce originali, dove ogni traccia è una lista di stringhe.

    Ritorna:
    - Similarità calcolata come 1 - CFld metric.
    """
    generated_traces = [[activity.name for activity in particle] for particle in final_particles]
    # Crea la matrice log per le tracce generate
    generated_log_matrix = generate_log_matrix(generated_traces, label_map)
    # Crea la matrice log per le tracce originali
    original_log_matrix = generate_log_matrix(original_traces, label_map)
    # Calcola la similarità confrontando il log originale con quello generato
    similarity = get_log_similarity(original_log_matrix, generated_log_matrix)
    return similarity


def _compute_pair_distances(original_log, generated_log):
    distances = []
    for i, original_trace in tqdm(enumerate(original_log), total=len(original_log), desc="Calcolo distanze"):
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


def get_log_similarity(original_log, generated_log, parallel=True, n_jobs=-1):
    if parallel:
        normalized_distances = _compute_pair_distances_parallel(original_log, generated_log, n_jobs)
    else:
        normalized_distances = _compute_pair_distances(original_log, generated_log)  # versione già esistente
    cost_matrix, row_ind, col_ind = _pair_traces(normalized_distances, original_log, generated_log)
    cfld_metric = _compute_cfld(row_ind, col_ind, cost_matrix)
    return 1 - cfld_metric

