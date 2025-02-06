import torch.nn as nn


class BertOutputClassificationHead(nn.Module):

    def __init__(self, model, output_size):
        super(BertOutputClassificationHead, self).__init__()
        self.model = model
        self.output_layer = nn.Linear(model.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # Prendi il primo token [CLS]
        return self.output_layer(last_hidden_state)  # Passa all'output layer

