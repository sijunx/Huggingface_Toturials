from flask import Flask, request, json

from my_txt_classify.myPredict import my_predict

app = Flask(__name__)

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
    out = my_predict(s1)
    return_dict['result'] = out
    return json.dumps(return_dict, ensure_ascii=False)

app.run(debug=False)
# if __name__ == '__main__':
#     print("Loading PyTorch model and Flask starting server ...")
#     print("Please wait until server has fully started")
#
#     app.run(debug=True)
