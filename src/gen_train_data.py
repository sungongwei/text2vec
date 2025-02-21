import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import random
import json
import time
import faiss
import os
import logging

from read_data import merge_json_files
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

d = 768
def filter_duplicate_dicts(dict_list, key_combo=('text1', 'text2')):
    """
    过滤包含重复指定键值对组合的字典。

    参数:
    dict_list (list of dict): 要过滤的字典列表。
    key_combo (tuple): 用于判断字典是否重复的键值对组合。

    返回:
    list of dict: 过滤后的字典列表。
    """
    seen_combos = set()
    result = []

    for d in dict_list:
        combo = tuple(d[k] for k in key_combo)
        if combo not in seen_combos:
            seen_combos.add(combo)
            result.append(d)

    return result
def cal_token(text):
    return len(tokenizer.tokenize(text))

def cal_similarity(distance):
    return  round(1 - distance / d, 4)

def similarity_to_distance(similar):
    return d * (1 - similar)
def vectorize():
    data=[]
    index = faiss.IndexFlatL2(d)
    index_with_ids = faiss.IndexIDMap(index)
    # 1. 准备数据
    # 2. 对问题进行编码和向量化
    start = time.time()
    question_length = len(question_list)
    
    for idx, questions in enumerate(question_list):
        vectors = []
        logging.info(f"向量化: {idx+1}/{question_length}")
        sub_len = len(questions["question"])
        for question in questions["question"]:
            if(sub_len>1 ):
                for i in range(sub_len - 1):  # 循环到倒数第二个元素
                    question2 =questions["question"][i + 1]
                    if question == question2:
                        continue
                    data.append({"text1": question, "text2": question2, "label":5 })
                    # data.append(f"{question}	{question2}	5\n")
                    # file.write(f"{question}	{question2}	1\n")
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
        min_distance, index = index_with_ids.search(np.vstack(vectors), 10)
        for i in range(len(index[0])):
            simi = cal_similarity(min_distance[0][i])
            if simi > 0.925:
                for  question in questions["question"]:
                    for  question2 in question_list[index[0][i]]['question']:
                        if question == question2:
                            continue
                        label =0
                        if simi > 0.975:
                            label = 3
                        elif simi > 0.95:
                            label = 2
                        else:
                            label = 1
                        data.append({"text1": question, "text2": question2, "label": label})
                          
                        # data.append(f"{question}	{question2}	{ 2 if cal_similarity(min_distance[0][i]) > 0.8 else 1}\n")
                # file.write(f"{question}	{question2}	0\n")
        index_with_ids.add_with_ids(np.vstack(vectors), np.full(len(vectors), idx))
    end = time.time()
    logging.info("\n向量化:{} seconds".format(end - start))

    logging.info(index_with_ids.ntotal)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    with open('train/all.jsonl', 'w', encoding='utf-8') as train_file:
        filtered_data = filter_duplicate_dicts(data)
        for text in filtered_data:
            train_file.write(json.dumps(text, ensure_ascii=False) + '\n')
    # with open('train/train.data', 'w', encoding='utf-8') as train_file:
    #     sublist = data[:int(len(data) * 0.7)]
    #     for text in sublist:
    #         train_file.write(text)

    #     train_file.close()
    # with open('train/test.data', 'w', encoding='utf-8') as test_file:
    #     sublist = data[int(len(data) * 0.7):int(len(data) * 0.85)]
    #     for text in sublist:
    #         test_file.write(text)
    #     test_file.close()
    # with open('train/vaild.data', 'w', encoding='utf-8') as vaild_file:
    #     sublist = data[int(len(data) * 0.85):]
    #     for text in sublist:
    #         vaild_file.write(text)
    #     vaild_file.close()
    # file.close()
    return index_with_ids


vectorize()


