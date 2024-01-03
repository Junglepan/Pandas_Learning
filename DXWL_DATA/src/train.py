import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_dataset,train,load_data_npy
from model import MLSTMfcn
from constants import NUM_CLASSES, MAX_SEQ_LEN, NUM_FEATURES

def main():
    dataset = args.dataset
    file_path_name = args.file_path_npy
    assert dataset in NUM_CLASSES.keys()

    load_data_npy(file_path_name,dataset)

    path = "../"+dataset+"/"
    train_dataset, test_dataset, val_dataset = load_dataset(file_path=path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    mlstm_fcn_model = MLSTMfcn(num_classes=NUM_CLASSES[dataset], 
                               max_seq_len=MAX_SEQ_LEN[dataset], 
                               num_features=NUM_FEATURES[dataset])
    mlstm_fcn_model.to(device)

    optimizer = optim.SGD(mlstm_fcn_model.parameters(), lr=args.learning_rate, momentum=0.9)
    criterion = nn.NLLLoss()

    train(mlstm_fcn_model, train_loader, val_loader, 
          criterion, optimizer, 
          epochs=args.epochs, print_every=100, device=device, run_name=args.name)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--name", type=str, default="model_mlstm_fcn")
    p.add_argument("--dataset", type=str, default='dxwl_dataset')
    p.add_argument("--file_path_npy",type=str,default='behavior_array')
    args = p.parse_args()
    main()
