import torch

def evaluate_model(model, data_loader, criterion, device, epoch=None, total_epochs=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # disattiva il calcolo del gradiente
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)  # classi predette per ogni esempio
            correct += (predictions == labels).sum().item()  # conta le previsioni corrette
            total += labels.size(0)  # numero di esempi nel batch

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0

    if epoch is not None and total_epochs is not None:
        print(f"Epoch {epoch}/{total_epochs} - Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    else:
        print(f"\nEvaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy
