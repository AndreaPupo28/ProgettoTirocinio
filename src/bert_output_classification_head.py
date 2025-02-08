import torch.nn as nn


class BertClassifier(nn.Module):
    def __init__(self, model, output_size, dropout_rate=0.3):
        super(BertClassifier, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout_rate)  # Aggiunto dropout
        self.output_layer = nn.Linear(model.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Alcuni modelli di BERT restituiscono direttamente logits, quindi controlliamo
        if hasattr(outputs, "logits"):
            logits = outputs.logits  # Se il modello ha già logits, li usiamo direttamente
        else:
            last_hidden_state = outputs.last_hidden_state[:, 0, :]
            logits = self.output_layer(self.dropout(last_hidden_state))  # Dropout prima della classificazione

        return logits  # Nessuna attivazione, perché CrossEntropyLoss si aspetta logits grezzi
