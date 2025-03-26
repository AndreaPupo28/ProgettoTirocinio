import torch


def train(model, train_loader, test_loader, optimizer, epochs, criterion, device, patience=3):
    model.train()
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} → Training Loss: {avg_train_loss:.4f}")

        # Calcolo della loss sul test set
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}/{epochs} → Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Ripristina la modalità training per la prossima epoca
        model.train()

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping attivato: la loss non sta migliorando.")
                break

    return model
