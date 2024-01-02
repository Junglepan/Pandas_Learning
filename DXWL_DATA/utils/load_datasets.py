import numpy as np
import torch.utils.data
from dataset import TensorDataset
file_path = '../dxwl_dataset/'

def load_data(file_name):
    X_train = np.load(file_name + "X_train.npy")
    X_test = np.load(file_name + "X_test.npy")
    X_val = np.load(file_name + "X_val.npy")
    return X_train, X_test, X_val

def load_label(file_name):
    y_train = np.load(file_name + "y_train.npy")
    y_test = np.load(file_name + "y_test.npy")
    y_val = np.load(file_name + "y_val.npy")
    return y_train, y_test, y_val


a,b,c = load_label(file_path)
x,y,z = load_data(file_path)
# print(a.shape)
# print(x.shape)


X_train = torch.utils.data.DataLoader(TensorDataset(x,a),shuffle=False)
print(type(X_train))
print(X_train.dataset)