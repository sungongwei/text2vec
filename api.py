from flask import Flask, request, jsonify
from src.base import answer_question

app = Flask(__name__)

@app.route('/q/<q>')
def qa(q):
    # 获取 POST 请求中的数据
    # data = request.json
    answer = answer_question(q)
    # 返回答案
    return answer

if __name__ == '__main__':
    app.run(debug=True,port=8000)
    print('Server running on port 8000...')
