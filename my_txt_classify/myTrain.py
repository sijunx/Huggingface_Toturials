import torch
from transformers import BertTokenizer
from transformers import AdamW

from my_txt_classify.MyDataset import MyDataset
from my_txt_classify.MyModel import MyModel

myTokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    # 编码
    data = myTokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                     truncation=True,
                                     padding='max_length',
                                     max_length=500,
                                     return_tensors='pt',
                                     return_length=True)

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    # print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels


# 数据加载器
my_dataset = MyDataset('train')
my_data_loader = torch.utils.data.DataLoader(dataset=my_dataset,
                                             batch_size=16,
                                             collate_fn=collate_fn,
                                             shuffle=True,
                                             drop_last=True)

my_model = MyModel()
# 训练
optimizer = AdamW(my_model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

my_model.train()
for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(my_data_loader):
    out = my_model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   token_type_ids=token_type_ids)

    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 5 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)

        print(i, loss.item(), accuracy)

    if i == 5:
        break

# 模型保存
# 使用transformers预训练后进行保存
model_path = '/Users/zard/Documents/nlp002/Huggingface_Toturials/data/model/model02.pkl'

# 预训练模型使用 `from_pretrained()` 重新加载
# my_model.from_pretrained(model_path)
# my_tokenizer.from_pretrained(tokenizer_path)
# 模型加载

# 模型预测

torch.save(my_model.state_dict(), model_path)
model = MyModel()
model.load_state_dict(torch.load(model_path))
model.eval()
