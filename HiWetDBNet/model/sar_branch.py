import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.convLSTM import ConvLSTM


class SARBranch(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_class2,
                 num_class3,
                 kernel_size,
                 num_layers,
                 batch_first=False,
                 bias=True,
                 return_all_layers=False
                 ):
        super(SARBranch, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.ConvLSTM = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=self.batch_first,
            bias=self.bias,
            return_all_layers=self.return_all_layers
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(128 * 7 * 7, num_class2)
        self.fc3 = nn.Linear(256 * 4 * 4, num_class3)

        self.level2_conv = nn.Conv2d(self.hidden_dim[0], 128, kernel_size=5, stride=2, padding=1)
        self.level3_conv_1 = nn.Conv2d(self.hidden_dim[-1], 256, kernel_size=5, stride=2, padding=1)
        self.level3_conv_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, sar_data):
        rnn_output, rnn_state = self.ConvLSTM(sar_data)  # LSTM
        output_level2 = rnn_output[0][:, -1, :, :, :]  # 15*15
        output_level2 = self.level2_conv(output_level2)  # 7*7
        output_level3 = rnn_output[-1][:, -1, :, :, :]  # 15*15
        output_level3 = self.level3_conv_1(output_level3)
        output_level3 = self.level3_conv_2(output_level3)
        output_level2_connect = torch.flatten(output_level2, 1)  # 6272
        output_level3_connect = torch.flatten(output_level3, 1)  # 4096

        output_level2 = self.fc2(output_level2_connect)
        output_level3 = self.fc3(output_level3_connect)

        return output_level2, output_level2_connect, output_level3, output_level3_connect


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SARBranch(input_dim=4,
                      hidden_dim=[64, 128],
                      num_class2=4,
                      num_class3=7,
                      kernel_size=(3, 3),
                      num_layers=2,
                      batch_first=True,
                      bias=True,
                      return_all_layers=True)
    sar_input = Variable(torch.randn(64, 15, 4, 15, 15))
    output_level2, output_level2_connect, output_level3, output_level3_connect = model(sar_input)
    print(output_level2.shape)
    print(output_level3.shape)
    print(output_level2_connect.shape)
    print(output_level3_connect.shape)
