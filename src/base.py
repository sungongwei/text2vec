import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import json
import time
import faiss
import os
from datetime import datetime
from src.read_data import merge_json_files

# 打开 JSON 文件并读取数据
with open("config.json") as f:
    config = json.load(f)
question_list = merge_json_files("./data")
if os.path.exists("index") == False:
    os.mkdir('index')


print(f"\r加载模型...", end="", flush=True)

# 初始化Bert模型和tokenizer
start = time.time()
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("./model")
model = BertModel.from_pretrained("./model").to(device)
model.eval()
end = time.time()
print("\n加载模型完成:", end - start, "seconds")


d = 768

def generate_index_path():
    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 将时间戳添加到基础路径中
    return f"index/{current_time}.index"

def vectorize(data):
    index = faiss.IndexFlatL2(d)
    global index_with_ids
    index_with_ids = faiss.IndexIDMap(index)
    # 1. 准备数据
    # 2. 对问题进行编码和向量化
    start = time.time()
    question_length = len(data)
    for idx, questions in enumerate(data):
        vectors = []
        print(f"\r向量化: {idx+1}/{question_length}", end="", flush=True)
        for question in questions["question"]:
            inputs = tokenizer(
                question, return_tensors="pt", max_length=512, padding=True, truncation=True
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                vector = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                vectors.append(vector)
        index_with_ids.add_with_ids(np.vstack(vectors), np.full(len(vectors), idx))
    end = time.time()

    print("\n向量化:", end - start, "seconds")

    print(index_with_ids.ntotal)
    faiss.write_index(index_with_ids, index_path)
    faiss.write_index(index_with_ids, generate_index_path())
                      
    global question_list
    question_list= data
    print(question_list)
    return index_with_ids

index_path = "faiss.index"
index_with_ids = None
# 检查索引文件是否存在
if os.path.exists(index_path):
    # 如果文件存在，则加载本地索引
    print("Loading index from file...")
    index_with_ids = faiss.read_index(index_path)
else:
    # 如果文件不存在，则创建一个新的索引
    print("Creating a new index...")
    vectorize(question_list)
    print("Saving index to file...")
    



def answer_question(user_input):
    print(index_with_ids.ntotal)
    user_inputs = tokenizer(
        user_input, return_tensors="pt", max_length=512, padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        user_outputs = model(**user_inputs)
        user_vector = torch.mean(user_outputs.last_hidden_state, dim=1).cpu().numpy()

    start = time.time()
    best_distance, best_index = index_with_ids.search(np.vstack(user_vector), 1)
    end = time.time()
    similar = round(1 - best_distance[0][0] / d, 4)
    print(similar,f"{end-start:.4f}")
    if similar < config["noAnswerThreshold"]:
        return config["noAnswerReply"]
    return question_list[best_index[0][0]]["answer"]
