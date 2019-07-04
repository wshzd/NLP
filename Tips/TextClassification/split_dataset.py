from sklearn.model_selection import train_test_split
import numpy as np


# 训练集和测试集生成
train_X, test_X, train_y, test_y = train_test_split(all_data, all_labels, test_size=0.2, random_state=10)

# 乱序训练集
index = [i for i in range(len(train_X))]
np.random.shuffle(index)
train_X = np.array(train_X)[index].tolist()
train_y = np.array(train_y)[index].tolist()

# 训练集batch划分
idx = 0
batch_idx = 0
tmp_batch_x = []
tmp_batch_y = []
fout = open(BATCH_PATH + str(batch_idx), "w", encoding="utf8")
while idx < len(train_X):
    fout.write("%s\t%s\n" % (train_y[idx], "\t".join(train_X[idx])))
    idx = idx + 1    
    if idx % BATCH_SIZE == 0:
        fout.close()
        batch_idx = batch_idx + 1
        fout = open(BATCH_PATH + str(batch_idx), "w", encoding="utf8")
fout.close()

# 测试集处理
index = [i for i in range(len(test_X))]
np.random.shuffle(index)
test_X = np.array(test_X)[index].tolist()
test_y = np.array(test_y)[index].tolist()
fout = open(BATCH_PATH + "test", "w", encoding="utf8")
for idx in range(len(test_X)):
    fout.write("%s\t%s\n" % (test_y[idx], "\t".join(test_X[idx])))
