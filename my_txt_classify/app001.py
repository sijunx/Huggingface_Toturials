# save this as app001.py
import datetime
import torch

from flask import Flask, request, json

app = Flask(__name__)

@app.route("/test", methods=["GET"])
def hello():
    #默认返回内容
    return_dict = {'return_code':'200','return_info':'处理成功','result':None}

    # 判断传入的json数据是否为空
    if len(request.get_data()) == 0:
        return_dict['return_code'] = '5004'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)



    return json.dumps(return_dict,ensure_ascii=False)




if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")

    app.run(debug=True)
