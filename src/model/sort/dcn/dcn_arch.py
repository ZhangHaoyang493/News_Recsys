import torch
import torch.nn as nn
import torch.nn.functional as F

class DCNLayer(nn.Module):
    def __init__(self, dim=32):
        super(DCNLayer, self).__init__()

        self.w = nn.Parameter(torch.empty(dim, 1))
        self.b = nn.Parameter(torch.zeros(dim, 1))

        nn.init.xavier_uniform_(self.w)

    def forward(self, x_l, x_0):
        """Deep & Cross Network forward pass.

        Args:
            x_l (_type_): Bxdim tensor, input feature at layer l
            x_0 (_type_): Bxdim tensor, original input feature
        """

        x_l = x_l.unsqueeze(-1)  # Bxdimx1
        x_0 = x_0.unsqueeze(-1)  # Bxdimx1

        cross = torch.matmul(torch.matmul(x_0, x_l.transpose(1, 2)), self.w)  # Bxdimx1

        out = cross + self.b + x_l  # Bxdimx1
        out = out.squeeze(-1)  # Bxdim

        return out


class DCNv2Layer(nn.Module):
    def __init__(self, dim=32):
        super(DCNv2Layer, self).__init__()

        self.linear = nn.Linear(dim, dim, bias=True)

    def forward(self, x_l, x_0):
        """Deep & Cross Network v2 forward pass.

        Args:
            x_l (_type_): Bxdim tensor, input feature at layer l
            x_0 (_type_): Bxdim tensor, original input feature
        """

        cross = x_0 * self.linear(x_l)  # Bxdim
        out = cross + x_l  # Bxdim

        return out
    

class DCNNet(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super(DCNNet, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(DCNLayer(input_dim))

        self.cross_net = nn.ModuleList(layers)

    def forward(self, x):
        x_0 = x
        for layer in self.cross_net:
            if isinstance(layer, DCNLayer):
                x = layer(x, x_0)
            else:
                x = layer(x)
        return x
    

class DCNv2Net(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super(DCNv2Net, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(DCNv2Layer(input_dim))
            layers.append(nn.ReLU())

        self.cross_net = nn.ModuleList(layers)

    def forward(self, x):
        x_0 = x
        for layer in self.cross_net:
            if isinstance(layer, DCNv2Layer):
                x = layer(x, x_0)
            else:
                x = layer(x)
        return x