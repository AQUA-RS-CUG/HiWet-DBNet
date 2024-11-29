import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.resnet import resnet_backbone


class OpticalBranch(nn.Module):
    def __init__(self,
                 num_class2: int = 4,
                 num_class3: int = 7
                 ):
        super(OpticalBranch, self).__init__()
        self.resnet = resnet_backbone(
            pretrained=False,
            progress=True,
            num_class2=num_class2,
            num_class3=num_class3
        )
        # self.fc2 = nn.Linear(128, num_class2)
        # self.fc3 = nn.Linear(256, num_class3)

    def forward(self, optical_data):
        out_level2, out_level2_connect, out_level3, out_level3_connect = self.resnet(optical_data)
        # out_level2 = self.fc2(out_level2)
        # out_level3 = self.fc3(out_level3)
        return out_level2, out_level2_connect, out_level3, out_level3_connect


if __name__ == "__main__":
    model = OpticalBranch(4, 7)
    sar_input = Variable(torch.randn(64, 12, 15, 15))
    out_level2, out_level2_connect, out_level3, out_level3_connect = model(sar_input)
    print(out_level2.shape)
    print(out_level2_connect.shape)
    print(out_level3.shape)
    print(out_level3_connect.shape)
