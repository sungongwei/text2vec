import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import json
import time
import faiss
import os
import logging

from src.read_data import merge_json_files
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 打开 JSON 文件并读取数据
with open("config.json") as f:
    config = json.load(f)
question_list = merge_json_files("./data")

logging.info(f"加载模型...")

# 初始化Bert模型和tokenizer
start = time.time()
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("./sentence_model")
model = BertModel.from_pretrained("./sentence_model").to(device)
model.eval()
end = time.time()
logging.info("加载模型完成:{}".format(end - start))

# d = 768
print( model.config.hidden_size)
d = model.config.hidden_size
def cal_token(text):
    return len(tokenizer.tokenize(text))

def cal_similarity(distance):
    return  round(1 - distance / d, 4)

def vectorize():
    index = faiss.IndexFlatL2(d)
    index_with_ids = faiss.IndexIDMap(index)
    # 1. 准备数据
    # 2. 对问题进行编码和向量化
    start = time.time()
    question_length = len(question_list)
    duplicate_list = []
    
    for idx, questions in enumerate(question_list):
        vectors = []
        logging.info(f"向量化: {idx+1}/{question_length}")
        for question in questions["question"]:
            inputs = tokenizer(
                question,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True,
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                vector = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                vectors.append(vector)
        min_distance, index = index_with_ids.search(np.vstack(vectors), 1)
        if cal_similarity(min_distance[0][0]) > 0.93:
            duplicate_list.append([cal_similarity(min_distance[0][0]),question,questions['reporter'],{question_list[index[0][0]]['question'][0]},question_list[index[0][0]]['reporter']])
          
          # logging.error(f"太过相似: {cal_similarity(min_distance[0][0])}:{index[0][0]}:{question}:{question_list[index[0][0]]['question'][0]}")
        index_with_ids.add_with_ids(np.vstack(vectors), np.full(len(vectors), idx))
        
        
    end = time.time()

    logging.info("\n向量化:{} seconds".format(end - start))
    duplicate_list.sort(key=lambda x: x[0], reverse=True)
    with open("./duplicate.jsonl", "w") as f:
        for duplicate in duplicate_list:
            f.write(str(duplicate) + "\n")
        f.close()

    logging.info(index_with_ids.ntotal)
    return index_with_ids


index_path = "faiss.index"
# 检查索引文件是否存在
if os.path.exists(index_path):
    # 如果文件存在，则加载本地索引
    logging.info("Loading index from file...")
    index_with_ids = faiss.read_index(index_path)
else:
    # 如果文件不存在，则创建一个新的索引
    logging.info("Creating a new index...")
    index_with_ids = vectorize()
    logging.info("Saving index to file...")
    faiss.write_index(index_with_ids, index_path)


def similarity_to_distance(similar):
    return d * (1 - similar)
def answer_question(user_input):
    start = time.time()
    user_inputs = tokenizer(
        user_input, return_tensors="pt", max_length=512, padding=True, truncation=True
    ).to(device)
    # logging.info("向量化:{} seconds".format(user_inputs))
    with torch.no_grad():
        user_outputs = model(**user_inputs)
        # logging.info("向量化:{} seconds".format(user_outputs))
        user_vector = torch.mean(user_outputs.last_hidden_state, dim=1).cpu().numpy()
        # logging.info("向量化:{} seconds".format(user_vector.flatten()))
        
    end1 = time.time()

    best_distance, best_index = index_with_ids.search(np.vstack(user_vector), 1)
    # for(distance, index) in zip(best_distance[0], best_index[0]):
    #     logging.info(f"搜索: {cal_similarity(distance)}:{index}")
    end = time.time()
    similar =cal_similarity(best_distance[0][0])
    res = ""
    if similar < config["noAnswerThreshold"]:
        res = config["noAnswerReply"]
    else:
        res = question_list[best_index[0][0]]["answer"]
    logging.info('<q>{}<q>{}<q>{}'.format(similar,user_input,res))
    return res
