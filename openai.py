from datetime import datetime
import uuid
import json

from flask import Flask, request, jsonify, Response
from src.base import answer_question,cal_token
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


def generate_response(messages):
    """生成响应内容"""
    # 模拟响应配置
    response = {
        "id": "chat",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "chat",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "这是一个模拟响应"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }
    response["id"] = f"chat-{uuid.uuid4().hex}"
    response["created"] = int(datetime.now().timestamp())
    response["choices"][0]["message"]["content"] = answer_question( messages[-1]["content"])
    response["usage"]["prompt_tokens"] = cal_token(messages[-1]["content"])
    response["usage"]["completion_tokens"] = cal_token(response["choices"][0]["message"]["content"])
    response["usage"]["total_tokens"] = response["usage"]["prompt_tokens"] + response["usage"]["completion_tokens"]
    return response

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI 风格聊天补全接口"""
    # 验证授权头
    # auth_header = request.headers.get('Authorization')
    # if auth_header and not auth_header.startswith('Bearer '):
    #     return jsonify({"error": "Invalid authorization header"}), 401
    # 获取请求参数
    data = request.get_json()
    # model = data.get('model', 'gpt-3.5-turbo')
    messages = data.get('messages', [])
    # 简单验证
    if not messages:
        return jsonify({"error": "messages is required"}), 400
    try:
        response_data = generate_response(messages)
        return Response(json.dumps(response_data), mimetype='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0",port=8000)
    print('Server running on port 8000...')
