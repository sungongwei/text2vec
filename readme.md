

## model

git clone https://www.modelscope.cn/Jerry0/text2vec-base-chinese.git  model   


## data
 问题答案目录
### 格式
 ```
  [{
    "question": ["问题"],
    "answer": "答案"
  }]
 ```
# CPU-only version
$ conda install -c pytorch faiss-cpu=1.9.0

# GPU(+CPU) version
$ conda install -c pytorch -c nvidia faiss-gpu=1.9.0
