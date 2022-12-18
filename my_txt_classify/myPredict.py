import torch
import time
from transformers import BertTokenizer

from my_txt_classify.MyModel import MyModel

my_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# model_path = '/Users/zard/Documents/nlp002/Huggingface_Toturials/data/model/model02.pkl'
model_path = '/Users/zard/Documents/nlp002/Huggingface_Toturials/data/model/model_my_006.pkl'
my_model = MyModel()

print("加载模型---------------------------------开始-------------")
my_model.load_state_dict(torch.load(model_path))
print("加载模型---------------------------------结束-------------")

def my_predict(s1):
    # 批量编码句子
    # s1 = '测试数据，情绪管理大师'
    # s2 = '我喜欢吃什么，苹果香蕉'
    # sents = [s1, s2]
    sents = [s1]
    start01 = time.time()
    out = my_tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                         truncation=True,
                                         padding='max_length',
                                         max_length=500,
                                         return_tensors='pt',
                                         return_length=True)
    end01 = time.time()
    print("编码耗时:", (end01 - start01))

    time_start = time.time()  # 记录开始时间
    result = my_model(input_ids=out['input_ids'],
                      attention_mask=out['attention_mask'],
                      token_type_ids=out['token_type_ids'])
    time_end = time.time()  # 记录开始时间

    print("模型预测耗时:", (time_end- time_start))

    print("result:", result)
    result = result.argmax(dim=1)
    print("result:", result)

    print("label:", result[0].item())
    return result[0].item()

txt01 = "测试数据，喜欢吃什么，苹果吗？"
out = my_predict(txt01)
print(out)
