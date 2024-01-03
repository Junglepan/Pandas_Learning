import numpy as np
import torch.utils.data
# from dataset import TensorDataset
import torch

from sklearn.model_selection import train_test_split
import os

def load_data_npy(file_path_npy,dataset):
    data = np.load('../'+dataset+'/'+file_path_npy+'.npy', allow_pickle=True)
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

def load_data(file_name):
    X_train = np.load(file_name + "X_train.npy")
    X_test = np.load(file_name + "X_test.npy")
    X_val = np.load(file_name + "X_val.npy")
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    X_val = torch.from_numpy(X_val)

    return X_train, X_test, X_val

def load_label(file_name):
    y_train = np.load(file_name + "y_train.npy")
    y_test = np.load(file_name + "y_test.npy")
    y_val = np.load(file_name + "y_val.npy")
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    y_val = torch.from_numpy(y_val).long()
    return y_train, y_test, y_val

def load_seq_lens(file_name):
    train_seq_lens = np.load(file_name + "seq_lens_train.npy")
    test_seq_lens = np.load(file_name + "seq_lens_test.npy")
    val_seq_lens = np.load(file_name + "seq_lens_val.npy")
    train_seq_lens = torch.from_numpy(train_seq_lens)
    test_seq_lens = torch.from_numpy(test_seq_lens)
    val_seq_lens = torch.from_numpy(val_seq_lens)
    return train_seq_lens , test_seq_lens, val_seq_lens

def load_dataset(file_path):
    X_train, X_test, X_val = load_data(file_path)
    y_train, y_test, y_val = load_label(file_path)
    train_seq_lens, test_seq_lens, val_seq_lens = load_seq_lens(file_path)
    # print(X_train.shape,y_train.shape,train_seq_lens.shape)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, train_seq_lens)
    test_dataset = torch.utils.data.TensorDataset(X_test,y_test, test_seq_lens)
    val_dataset = torch.utils.data.TensorDataset(X_val,y_val, val_seq_lens)
    return train_dataset,test_dataset,val_dataset


def validation(model, testloader, criterion, device='cpu'):
    accuracy = 0
    test_loss = 0
    for inputs, labels, seq_lens in testloader:
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs, seq_lens)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def train(model, trainloader, validloader, criterion, optimizer,
          epochs=10, print_every=10, device='cpu', run_name='model_mlstm_fcn'):
    print("Training started on device: {}".format(device))
    valid_loss_min = np.Inf  # track change in validation loss
    steps = 0

    for e in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for inputs, labels, seq_lens in trainloader:
            steps += 1

            inputs = inputs.float()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs, seq_lens)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.6f}.. ".format(train_loss / print_every),
                      "Val Loss: {:.6f}.. ".format(valid_loss / len(validloader)),
                      "Val Accuracy: {:.2f}%".format(accuracy / len(validloader) * 100))

                # save model if validation loss has decreased
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        valid_loss))
                    os.makedirs("../weights",exist_ok=True)
                    torch.save(model.state_dict(), '../weights/' + run_name + '.pt')
                    valid_loss_min = valid_loss

                train_loss = 0

                model.train()




if __name__ == "__main__":
    file_path = '../dxwl_dataset/'
    # a, b, c = load_label(file_path)
    # x, y, z = load_data(file_path)
    # print(a.shape)
    # print(x.shape)
    train_dataset, test_dataset, val_dataset = load_dataset(file_path = file_path)
    X_train = torch.utils.data.DataLoader(train_dataset, shuffle=False)
    print(type(X_train))
    print(X_train.dataset)