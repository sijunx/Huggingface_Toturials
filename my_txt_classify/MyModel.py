import torch
from transformers import BertModel


class MyModel(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        # 加载预训练模型
        self.pretrained = BertModel.from_pretrained('bert-base-chinese')
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out
