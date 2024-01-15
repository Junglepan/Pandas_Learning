import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn.parameter import Parameter

class ECA_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3, gamma=2, b=1):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.gamma = gamma
        self.b = b


    def forward(self, x):
        # 计算通道维度上的注意力权重
        b, c, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(self.gamma * y + self.b)
        # 对输入应用注意力权重
        return x * y.expand_as(x)


class ECA_LSTM_NET(nn.Module):
    def __init__(self, *, num_classes, max_seq_len, num_features,
                 num_lstm_out=128, num_lstm_layers=1,
                 conv1_nf=128, conv2_nf=256, conv3_nf=128,
                 lstm_drop_p=0.8, fc_drop_p=0.3):
        super(ECA_LSTM_NET, self).__init__()
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
        self.lstm = nn.LSTM(input_size=self.num_features,
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)

        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        self.se1 = ECA_layer(1)  # ex 128
        self.se2 = ECA_layer(1)  # ex 256

        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.fc = nn.Linear(self.conv3_nf + self.num_lstm_out, self.num_classes)
        # self.fc = nn.Linear(self.num_lstm_out, self.num_classes)        # 只使用MLSTM部分的预测
        # self.fc = nn.Linear(self.conv3_nf, self.num_classes)        # 只使用FCN部分的预测

    def forward(self, x, seq_lens):
        ''' input x should be in size [B,T,F], where
            B = Batch size
            T = Time samples
            F = features
        '''

        # 处理时间序列长度不同的问题
        # x1 = nn.utils.rnn.pack_padded_sequence(x, seq_lens,
        #                                        batch_first=True,
        #                                        enforce_sorted=False)
        x1, _ = self.lstm(x)
        # x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True,
        #                                          padding_value=0.0)
        x1 = x1[:, -1, :]

        # FCN
        x2 = x.transpose(2, 1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)
        x_all = torch.cat((x1, x2), dim=1)

        # 只使用LSTM训练：
        # x_all = x1

        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out




if __name__ == "__main__":
    print("panbk")