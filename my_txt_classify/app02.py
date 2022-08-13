# from chatbot import CB
from flask import Flask, render_template, request, json
import time


app = Flask(__name__)

import torch
from transformers import BertTokenizer

from my_txt_classify.MyDataset import MyDataset
from my_txt_classify.MyModel import MyModel

my_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model_path = '/Users/zard/Documents/nlp002/Huggingface_Toturials/data/model/model02.pkl'
my_model = MyModel()
print("加载模型---------------------------------开始-------------")
my_model.load_state_dict(torch.load(model_path))
print("加载模型---------------------------------结束-------------")

@app.route("/test", methods=["GET"])
def hello():
    # 默认返回内容
    return_dict = {'return_code': '200', 'return_info': '处理成功', 'result': None}
    print("request数值：", request)
    print("request.get_data() 数值：", request.get_data())
    print("request.get_json() 数值：", request.get_json())
    # 判断传入的json数据是否为空
    if len(request.get_data()) == 0:
        return_dict['return_code'] = '5004'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)

    s1 = request.get_json()['msg']
    print("s1:", s1)
    # s2 = request.values.get('s2')
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
    out = result.argmax(dim=1)
    return_dict['result'] = out[0].item()
    return json.dumps(return_dict, ensure_ascii=False)


app.run(debug=False)
# if __name__ == '__main__':
#     print("Loading PyTorch model and Flask starting server ...")
#     print("Please wait until server has fully started")
#
#     app.run(debug=True)
