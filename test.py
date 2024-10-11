import faiss
import time
import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 1                      # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
print(xq[0])
xq[:, 0] += np.arange(nq) / 1000.
# 假设 question_vectors 是一个 numpy array，形状为 (num_questions, vector_dim)
index = faiss.IndexFlatL2(d)   # build the index
# 包装为 IndexIDMap 来添加自定义 ID
index_with_ids = faiss.IndexIDMap(index)

# 创建一个自定义 ID 列表
custom_ids = np.arange(1000, 1000 + nb).astype('int64')

# 将数据添加到带有自定义 ID 的索引中
index_with_ids.add_with_ids(xb, custom_ids)
print(index_with_ids.is_trained)        # IndexFlatL2不需要训练过程
print(index_with_ids.ntotal)

print(xq[0])

k = 4      
start = time.time()# we want to see 4 nearest neighbors
D, I = index_with_ids.search(xq, k)     # search
end = time.time()
print(end-start,)
print(I[:5])                   # neighbors of the 5 first queries
print(D[:5])   