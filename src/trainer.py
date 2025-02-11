import torch
import numpy as np

def train(model, train_data_loader, test_data_loader, optimizer, epochs, criterion, device, patience=3):
    model.train()
    best_loss = float("inf")
    patience_counter = 0  # Contatore per Early Stopping

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_data_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_data_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_data_loader)
        print(f"Epoch {epoch + 1}/{epochs} completato - Loss media: {avg_loss:.4f}")

        # Early Stopping: se la loss non migliora per "patience" epoche, ferma il training
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # Reset del contatore
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping attivato: nessun miglioramento della loss per", patience, "epoche consecutive.")
                break  # Ferma il training

    return model


import numpy as np

def predict_next_log(model, tokenizer, current_log, label_map, device, num_particles=100):
    model.eval()  # Imposta il modello in modalità valutazione
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

    #Stampa tutti i possibili log successivi con le probabilità
    sorted_indices = np.argsort(probs)[::-1]  # Ordina le probabilità in ordine decrescente
    print("\n Possibili log successivi con probabilità:")
    for idx in sorted_indices[:10]:  # Mostra i primi 10 log più probabili
        log_name = list(label_map.keys())[idx]
        log_prob = probs[idx]
        print(f"  - {log_name}: {log_prob:.4f}")

    #Usa il Particle Filtering per determinare il log più probabile
    particles = np.random.choice(list(label_map.keys()), size=num_particles, p=probs)
    unique, counts = np.unique(particles, return_counts=True)
    most_likely = unique[np.argmax(counts)]

    return most_likely, probs
