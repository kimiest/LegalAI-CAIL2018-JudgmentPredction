from transformers import AutoModel
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768, 196)

    def forward(self, batch_inputs):
        out = self.bert(input_ids=batch_inputs['input_ids'].squeeze(),
                           attention_mask=batch_inputs['attention_mask'].squeeze()).pooler_output
        out = self.fc1(out)
        return out