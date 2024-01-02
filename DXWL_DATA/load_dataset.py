import torch
def load_datasets():
    data_path = './'+'dxwl_dataset'+'/'

    X_train = torch.load(data_path+'X_train.npy')
    X_val = torch.load(data_path+'X_val.npy')
    X_test = torch.load(data_path+'X_test.npy')

    y_train = torch.load(data_path+'y_train.npy')
    y_val = torch.load(data_path+'y_val.npy')
    y_test = torch.load(data_path+'y_test.npy')


    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    return train_dataset, val_dataset, test_dataset




