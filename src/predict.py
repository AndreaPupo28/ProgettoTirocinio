import torch
import numpy as np

def predict_next_log(model, tokenizer, current_log, label_map, device, num_particles=100):
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

    sorted_indices = np.argsort(probs)[::-1]
    for idx in sorted_indices[:10]:
        log_name = list(label_map.keys())[idx]
        log_prob = probs[idx]
        print(f"  - {log_name}: {log_prob:.4f}")

    particles = np.random.choice(list(label_map.keys()), size=num_particles, p=probs)
    unique, counts = np.unique(particles, return_counts=True)
    most_likely = unique[np.argmax(counts)]

    return most_likely, probs
