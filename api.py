from flask import Flask, request, jsonify
from src.base import answer_question ,vectorize
app = Flask(__name__)

@app.route('/q/<q>')
def qa(q):
    # 获取 POST 请求中的数据
    # data = request.json
    answer = answer_question(q)
    # 返回答案
    return answer
@app.route('/train/', methods=['POST'])
def vector():
    # 获取 POST 请求中的数据
    data = request.json
    vectorize(data)
    # 返回答案
    return jsonify({'code': 0})
if __name__ == '__main__':
    app.run(debug=False,port=8000)
    print('Server running on port 8000...')
