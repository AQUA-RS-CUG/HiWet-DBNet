import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.optical_branch import OpticalBranch
from model.sar_branch import SARBranch
# from optical_branch import OpticalBranch
# from sar_branch import SARBranch

class MSCNN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 num_class2, 
                 num_class3, 
                 kernel_size, 
                 num_layers, 
                 batch_first: bool = True, 
                 bias: bool = True, 
                 return_all_layers: bool = False, 
                 ):
        super(MSCNN, self).__init__()
        
        self.sar_model = SARBranch(
                    input_dim = input_dim, 
                    hidden_dim = hidden_dim, 
                    num_class2 = num_class2, 
                    num_class3 = num_class3, 
                    kernel_size = kernel_size, 
                    num_layers = num_layers,  
                    batch_first = batch_first, 
                    bias = bias, 
                    return_all_layers = return_all_layers)
        
        self.optical_model = OpticalBranch(
            num_class2 = num_class2, 
            num_class3 = num_class3
        )
        
        # self.fc2 = nn.Linear(128 * 2, num_class2)
        # self.fc3 = nn.Linear(256 * 2, num_class3)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # 不要avgpool
        self.fc2 = nn.Linear(128 * 7 * 7 * 2, num_class2)
        # self.fc2_2 = nn.Linear(128 * 7 * 7 * 2, 2048)
        # self.fc3 = nn.Linear(256 * 4 * 4 + 256 * 7 * 7, num_class3)
        self.fc3 = nn.Linear(256 * 4 * 4 * 2, num_class3)
        # self.fc3_1 = nn.Linear(256 * 4 * 4 * 2, 4096)
        # 实验性添加
        # self.fc2 = nn.Linear(64 * 13 * 13 + 128 * 7 * 7, num_class2)
        
    def forward(self, optical_data, sar_data):
        o_output2, o_connect2, o_output3, o_connect3 = self.optical_model(optical_data) ########
        _, s_connect2, _, s_connect3 = self.sar_model(sar_data)
        
        output_level2 = torch.cat((o_connect2, s_connect2), dim=1)
        output_level3 = torch.cat((o_connect3, s_connect3), dim=1)
        
        output_level2 = self.fc2(output_level2)
        output_level3 = self.fc3(output_level3)
        
        return output_level2, output_level3, o_output2, o_output3 #######
        
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MSCNN(input_dim=4, 
                    hidden_dim=[64, 128], 
                    num_class2=4, 
                    num_class3=6, 
                    kernel_size=(3,3), 
                    num_layers=2, 
                    batch_first=True, 
                    bias=True, 
                    return_all_layers=True)
    sar_input = Variable(torch.randn(64, 15, 4, 15, 15))
    optical_input = Variable(torch.randn(64, 10, 15, 15))
    output_level2, output_level3, s_output2, s_output3 = model(optical_input, sar_input)
    print(output_level2.shape)
    print(output_level3.shape)
    print(s_output2.shape)
    print(s_output3.shape)