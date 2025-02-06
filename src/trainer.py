import torch

def train(model, train_data_loader, test_data_loader, optimizer, epochs, criterion, device):
    model.train()  # Imposta il modello in modalità training

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_data_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()  # Reset del gradiente
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # Passa l'input al modello

            # Assicuriamoci che gli output siano logits per CrossEntropyLoss
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            loss = criterion(outputs, labels)  # Calcola la perdita
            loss.backward()  # Backpropagation
            optimizer.step()  # Aggiorna i pesi

            total_loss += loss.item()

            #  Mostra il progresso ogni 10 batch
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_data_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_data_loader)
        print(f" Epoch {epoch + 1}/{epochs} completato - Loss media: {avg_loss:.4f}")

    return model
