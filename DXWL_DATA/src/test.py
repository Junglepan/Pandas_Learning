import torch
import torch.nn as nn
import torch.optim as optim
from model import MLSTMfcn
from utils import validation, load_dataset
from constants import NUM_CLASSES, MAX_SEQ_LEN, NUM_FEATURES


def main():
    dataset = args.dataset
    assert dataset in NUM_CLASSES.keys()
    path = "../" + dataset + "/"
    _, _, test_dataset = load_dataset(file_path=path)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    mlstm_fcn_model = MLSTMfcn(num_classes=NUM_CLASSES[dataset], 
                               max_seq_len=MAX_SEQ_LEN[dataset], 
                               num_features=NUM_FEATURES[dataset])
    mlstm_fcn_model.load_state_dict(torch.load('../weights/'+args.weights))
    mlstm_fcn_model.to(device)

    criterion = nn.NLLLoss()
    test_loss, accuracy = validation(mlstm_fcn_model, test_loader, criterion, device)
    print("Test loss: {:.6f}.. Test Accuracy: {:.2f}%".format(test_loss, accuracy*100))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--weights", type=str, default="model_mlstm_fcn.pt")
    p.add_argument("--dataset", type=str, default="dxwl_dataset")
    args = p.parse_args()
    main()
