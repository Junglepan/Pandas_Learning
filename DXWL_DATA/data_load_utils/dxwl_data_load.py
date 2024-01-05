
import numpy as np
import torch.utils.data
import torch
from sklearn.model_selection import train_test_split
import os


def load_data_npy():
    data = np.load('../dxwl_dataset/behavior_array.npy', allow_pickle=True)
    data = np.array(data, dtype=np.float64)
    num_samples = data.shape[0]
    # 生成随机的五分类标签（0到4之间的整数）
    labels = np.random.randint(0, 5, size=num_samples)
    # print(labels.shape)

    indices = np.arange(len(data))
    # 划分数据集，test_size 指定测试集的比例
    train_indices, temp_indices = train_test_split(indices, test_size=0.4, random_state=42)
    # 继续划分，得到验证集和测试集的索引
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    # 获取训练集、验证集和测试集的数据和对应的索引
    X_train = data[train_indices]
    X_val = data[val_indices]
    X_test = data[test_indices]
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]

    seq_lens_train = np.asarray([i.shape[0] for i in X_train])
    seq_lens_test = np.asarray([i.shape[0] for i in X_test])
    seq_lens_val = np.asarray([i.shape[0] for i in X_val])


    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)
    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

    # print(type(X_train))
    os.makedirs('../dxwl_dataset', exist_ok=True)  # 即使存在也不报错
    file_name = '../dxwl_dataset'
    np.save(file_name + '/X_train', X_train)
    np.save(file_name + '/X_test', X_test)
    np.save(file_name + '/X_val', X_val)
    np.save(file_name + '/y_train', y_train)
    np.save(file_name + '/y_test', y_test)
    np.save(file_name + '/y_val', y_val)
    np.save(file_name + '/seq_lens_train',seq_lens_train)
    np.save(file_name + '/seq_lens_test', seq_lens_test)
    np.save(file_name + '/seq_lens_val', seq_lens_val)

    print(X_train.shape)
    print(y_train.shape)
    print(seq_lens_train.shape)
    print(seq_lens_train[0])

if __name__ =="__main__":
    load_data_npy()