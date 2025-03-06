import torch


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0
    print(f"\nEvaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    similarity_score = evaluate_log_similarity(model, data_loader.dataset, data_loader.dataset.label_map, device)
    print(f"CFld Similarity: {similarity_score:.4f}")
    return avg_loss, accuracy
