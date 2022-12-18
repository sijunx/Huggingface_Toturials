from transformers import BertTokenizer

myTokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 编码
txt01 = '数据'
result01 = myTokenizer.encode(text=txt01,
                                 truncation=True,
                                 padding='max_length',
                                 max_length=50,
                                 return_tensors='pt',
                                 return_length=True)

print("result01:", result01)
