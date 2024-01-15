import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = F.softmax(attention_weights, dim=-1)

        attended_values = torch.matmul(attention_weights, value)
        # output = torch.sum(attended_values, dim=1)
        output = attended_values
        # output = torch.cat([attended_values] * attended_values.size(1), dim=1)
        return output




class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)




class BiAMLSTMfcn(nn.Module):
    def __init__(self, num_classes, max_seq_len, num_features,
                 num_lstm_out=128, num_lstm_layers=1,
                 conv1_nf=128, conv2_nf=256, conv3_nf=256,
                 lstm_drop_p=0.8, fc_drop_p=0.3):

        super(BiAMLSTMfcn, self).__init__()
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_features = num_features

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers
        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf
        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        # Bi-LSTM Layer
        self.lstm = nn.LSTM(input_size=self.num_features,
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True,
                            bidirectional=True)

        # Attention Layer
        self.attention = AttentionModule(hidden_size=self.num_lstm_out * 2)  # Multiply by 2 for bidirectional LSTM

        # Convolutional Layers
        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        # SE Layers
        self.se1 = SELayer(self.conv1_nf)
        self.se2 = SELayer(self.conv2_nf)

        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        # Fully Connected Layer
        self.fc = nn.Linear(self.conv3_nf + self.num_lstm_out * 2, self.num_classes)

    def forward(self, x, seq_lens):
        # Bi-LSTM
        x1 = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        x1, _ = self.lstm(x1)
        x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True, padding_value=0.0)
        x1 = x1[:, -1, :]  # Take the output of the last time step

        # # Attention
        x1 = self.attention(x1)

        # Convolutional Layers
        x2 = x.transpose(2, 1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)
        # x2 = torch.mean(x2, dim=1).squeeze(1)  # Average pooling over the time dimension

        # print(x2.shape)
        # print(x1.shape)

        # Concatenate Bi-LSTM and Convolutional outputs
        x_all = torch.cat((x1, x2), dim=1)
        # Concatenate Bi-LSTM and Convolutional outputs
        # Concatenate the two tensors
        # Fully Connected Layer
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out




if __name__ == "__main__":
    from torchsummary import summary
    model = BiAMLSTMfcn(num_classes=19,
                               max_seq_len=45,
                               num_features=125)
    # print(model.state_dict())
    for  name,param in model.named_parameters():
        print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)