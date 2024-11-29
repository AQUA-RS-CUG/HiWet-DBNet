import torch.nn as nn
import torch
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
        hidden_dim: int
        kernel_size: (int, int)
        bias: bool
        """

        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim  # Number of input tensor channels
        self.hidden_dim = hidden_dim  # Number of hidden state channels
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # Convolutional outputs are guaranteed to be the same size
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              # The number of input channels is the sum of the number of input tensor channels and the number of hidden state channels.
                              out_channels=4 * self.hidden_dim,
                              # The unit extracts features by convolution includes 4 parts of features (i,f,o,g)
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        """
        Parameters
        ----------
        input_tensor:
            The input image itself
        cur_state:
            The hidden state tensor h_{t-1} and C_{t-1} passed from the previous convLSTM unit
        """
        h_cur, c_cur = cur_state  # The hidden state tensor h_{t-1} and C_{t-1} passed from the previous convLSTM unit
        # concatenate along channel axis (hide state tensor and input tensor superimposed in channel dimension)
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)  # 卷积提取特征
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim,
                                             dim=1)  # Split combined_conv by hidden_dim size in dim dimension
        # Compute the corresponding features in the convLSTM cell
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g  # Transfer to the hidden state of the next cell
        h_next = o * torch.tanh(c_next)  # Transfer to the hidden state of the next cell

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Parameters
        ----------
        batch_size:
            Image Batch Size
        image_size:
            Image Size
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
        # return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.device),
        #         torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # 为了储存每一层参数的尺寸
        cell_list = []  # 创建convLSTM单元list
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        # print(input_tensor.shape)
        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        # Stores the list of output data
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)  # 获取时间步长
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # Choose whether you want to output all the data or the last layer of data
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):  # 检查卷积核的格式是否为tuple或list
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    convlstm = ConvLSTM(input_dim=4, hidden_dim=[64, 128, 256], kernel_size=(3, 3), num_layers=3, batch_first=True,
                        return_all_layers=True)
    loss_fn = torch.nn.MSELoss()
    convlstm.to(device)

    """
    Parameters
    ----------
    b(batch size): number of batch samples in the input model
    t(time step): size of the time step in the model (equivalent to the number of convLSTM cells in the model)
    c(number of channels): number of channels in the input sample image (number of radar images in a month, or optical bands for optical images)
    h(number of rows): number of rows of the input sample image
    w(number of columns): number of columns of the input sample image
    """
    input = Variable(torch.randn(32, 10, 4, 15, 15)).to(device)
    target = Variable(torch.randn(1, 32, 64, 32)).double()

    layer_out, hidden_state = convlstm(input)
    print(len(layer_out))
    print(layer_out[0].shape)
    print(layer_out[1].shape)
    print(layer_out[2].shape)
