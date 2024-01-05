import numpy as np
import torch.utils.data
# from dataset import TensorDataset
import torch

from sklearn.model_selection import train_test_split
import os


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
    return train_dataset,val_dataset,test_dataset


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

    train_accuracies = []
    train_losses = []
    val_losses = []
    val_accuracies = []

    for e in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        train_acc_temp = 0



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

            # 统计分类正确的数目
            ps = torch.exp(output)
            equality = (labels.data == ps.max(1)[1])
            train_acc_temp += equality.type_as(torch.FloatTensor()).mean()


            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)


                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.6f}.. ".format(train_loss / print_every),
                      "Val Loss: {:.6f}.. ".format(valid_loss / len(validloader)),
                      "Val Accuracy: {:.2f}%".format(accuracy / len(validloader) * 100))

                # 统计指标
                train_accuracy = train_acc_temp / print_every
                avg_train_loss = train_loss / print_every
                val_accuracy = accuracy / len(validloader)
                val_loss = valid_loss / len(validloader)


                train_accuracies.append(train_accuracy)
                train_losses.append(avg_train_loss)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                # save model if validation loss has decreased
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        valid_loss))
                    os.makedirs("../weights",exist_ok=True)
                    torch.save(model.state_dict(), '../weights/' + run_name + '.pt')
                    valid_loss_min = valid_loss

                train_loss = 0
                train_acc_temp = 0;

                model.train()
# 绘制训练损失和验证损失的图表
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over ten batches')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制训练精度和验证精度的图表
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy over ten batches')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



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