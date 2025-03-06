from predict_parallel import predict_parallel_sequences

def predict_next_log_with_constraints(model, tokenizer, current_log, label_map, device, num_candidates):
    return predict_parallel_sequences(model, tokenizer, current_log, label_map, device, num_candidates)
