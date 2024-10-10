import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import time
from src.read_data import merge_json_files
# 打开 JSON 文件并读取数据
with open('config.json') as f:
    config = json.load(f)
question_list= merge_json_files('./data')


print(f"\r加载模型...", end="", flush=True)

# 初始化Bert模型和tokenizer
start = time.time()
if(torch.backends.mps.is_available()) :
  device = torch.device("mps")
elif(torch.cuda.is_available()):
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("./model")
model = BertModel.from_pretrained("./model").to(device)
model.eval()
end = time.time()
print("\n加载模型完成:", end - start, "seconds")

# 1. 准备数据
# 2. 对问题进行编码和向量化
start = time.time()
question_vectors = []
question_length = len(question_list)
for index,questions in enumerate(question_list):
  vectors = []
  print(f"\r向量化: {index+1}/{question_length}", end="", flush=True)
  for question in questions['question']:
    inputs = tokenizer(question, return_tensors="pt",max_length=512, padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        vector = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        # question_vector = torch.mean(outputs.last_hidden_state, dim=1)
        vectors.append(vector)
  question_vectors.append( np.vstack(vectors))
end = time.time()
# question_vectors_base = np.vstack(question_vectors)
# question_vectors_base = torch.vstack(question_vectors)
# print(question_vectors_base)
print("\n向量化:", end - start, "seconds")




def answer_question(user_input):
  start = time.time()
  user_inputs = tokenizer(user_input, return_tensors="pt",max_length=512, padding=True, truncation=True).to(device)
  with torch.no_grad():
      user_outputs = model(**user_inputs)
      user_vector = torch.mean(user_outputs.last_hidden_state, dim=1).cpu().numpy()
      # user_vector = torch.mean(user_outputs.last_hidden_state, dim=1)

  # 4. 计算相似度
  end = time.time()
  # print("用户输入向量化:",user_input,":", end - start, "seconds")

  start = time.time()
  # 计算输入文本与每个问题组的问题的相似度
  similarities = []
  for vectors in question_vectors:
      similaritie_score = cosine_similarity(user_vector, vectors)[0]
      max_similarity = np.max(similaritie_score)
      similarities.append(max_similarity)
  # similarities = torch.nn.functional.cosine_similarity(user_vector, question_vectors_base, dim=1).cpu().numpy()

  end = time.time()
  # print("计算相似度:", end - start, "seconds")

  best_match_index = np.argmax(similarities)
  if(similarities[best_match_index] < config['noAnswerThreshold']):
    return config['noAnswerReply']
  best_answer = question_list[best_match_index]["answer"]
  return best_answer