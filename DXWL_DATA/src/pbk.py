import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os

data = np.load('../dxwl_dataset/behavior_array.npy', allow_pickle=True)
print(data.shape)
print(type(data))
data = np.array(data,dtype=np.float64)
# 假设有 297 个样本
num_samples = data.shape[0]

# 生成随机的五分类标签（0到4之间的整数）
labels = np.random.randint(0, 5, size=num_samples)
# print(labels.shape)


indices = np.arange(len(data))
# 划分数据集，test_size 指定测试集的比例
train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
# 继续划分，得到验证集和测试集的索引
valid_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

# 获取训练集、验证集和测试集的数据和对应的索引
X_train = data[train_indices]
X_val = data[valid_indices]
X_test = data[test_indices]
y_train = labels[train_indices]
y_val = labels[valid_indices]
y_test = labels[test_indices]
print(train_indices.shape)
print(type(train_indices))
print(type(np.asarray([i.shape[0] for i in X_train])))
print(np.asarray([i.shape[0] for i in X_train]).shape)
# print(valid_indices)

# X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.3,random_state=0)
# X_test,X_val,y_test,y_val = train_test_split(X_test,y_test,test_size=0.5,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
#
# X_train = torch.from_numpy(X_train)
# y_train = torch.from_numpy(y_train)
# X_test = torch.from_numpy(X_test)
# y_test = torch.from_numpy(y_test)
# X_val = torch.from_numpy(X_val)
# y_val = torch.from_numpy(y_val)
# # print(type(X_train))
#
# os.makedirs('../dxwl_dataset', exist_ok=True)  # 即使存在也不报错
file_name = '../dxwl_dataset'
#
# np.save(file_name + '/X_train',X_train)
# np.save(file_name + '/X_test',X_test)
# np.save(file_name + '/X_val',X_val)
# np.save(file_name + '/y_train',y_train)
# np.save(file_name + '/y_test',y_test)
# np.save(file_name + '/y_val',y_val)

data1 = np.load(file_name + '/X_train.npy')

print(type(data1))


