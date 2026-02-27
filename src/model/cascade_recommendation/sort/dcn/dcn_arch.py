import torch
import torch.nn as nn
import torch.nn.functional as F

class DCNLayer(nn.Module):
    def __init__(self, dim=32):
        super(DCNLayer, self).__init__()
        self.w = nn.Parameter(torch.empty(dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x_l, x_0):
        """
        优化后的 Cross Layer
        x_l, x_0: [Batch, dim]
        """
        # 1. 先计算 x_l^T * w  (结果是标量，维度 [B, 1])
        # x_l: [B, d], w: [d, 1] -> dot: [B, 1]
        dot = torch.matmul(x_l, self.w) 
        
        # 2. 再乘 x_0
        # [B, d] * [B, 1] (广播机制) -> [B, d]
        cross = x_0 * dot
        
        return cross + self.b + x_l


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
        self.num_layers = num_layers
        self.layers = nn.ModuleList([DCNLayer(input_dim) for _ in range(num_layers)])

    def forward(self, x):
        x_0 = x
        x_l = x
        for layer in self.layers:
            x_l = layer(x_l, x_0)
        return x_l
    

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