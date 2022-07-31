import torch
from transformers import BertTokenizer

from my_txt_classify.MyDataset import MyDataset
from my_txt_classify.MyModel import MyModel

my_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model_path = '/Users/zard/Documents/nlp002/Huggingface_Toturials/data/model/model02.pkl'

# 批量编码句子
s1 = '测试数据，情绪管理大师'
s2 = '我喜欢吃什么，苹果香蕉'
sents = [s1, s2]
out = my_tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                     truncation=True,
                                     padding='max_length',
                                     max_length=500,
                                     return_tensors='pt',
                                     return_length=True)

my_model = MyModel()
my_model.load_state_dict(torch.load(model_path))

result = my_model(input_ids=out['input_ids'],
                  attention_mask=out['attention_mask'],
                  token_type_ids=out['token_type_ids'])

print("result:", result)
