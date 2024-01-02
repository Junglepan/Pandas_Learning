import torch
import numpy as np
from sklearn.model_selection import train_test_split  #数据分区
import os

data = np.load('./behavior_array.npy',allow_pickle=True)
print(data.shape)
print(type(data))
data = np.array(data,dtype=np.float64)
# 假设有 297 个样本
num_samples = 297

# 生成随机的五分类标签（0到4之间的整数）
labels = np.random.randint(0, 5, size=num_samples)
# print(labels.shape)

X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.3,random_state=0)
X_test,X_val,y_test,y_val = train_test_split(X_test,y_test,test_size=0.5,random_state=0)
# print(X_train.shape)
# print(X_test.shape)
# print(X_val.shape)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)
print(type(X_train))

os.makedirs('./dxwl_dataset',exist_ok=True)  # 即使存在也不报错
file_name = './dxwl_dataset'

np.save(file_name + '/X_train',X_train)
np.save(file_name + '/X_test',X_test)
np.save(file_name + '/X_val',X_val)
np.save(file_name + '/y_train',y_train)
np.save(file_name + '/y_test',y_test)
np.save(file_name + '/y_val',y_val)




