import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/ajb46717/workDir/projects/nsgaformer')

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    # 'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    # 'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#     'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
#     'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
#     'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
#     'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
#     'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
#     'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
#         nn.ReLU(inplace=False),
#         nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
#         nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
#         nn.BatchNorm2d(C, affine=affine)
#     ),
    
    'conv_1x1': lambda C, stride, affine: nn.Conv1d(C, C, 1, stride),
    'conv_3x1': lambda C, stride, affine: nn.Conv1d(C, C, 3, 1, bias=False),
    'sep_conv_3x1': lambda C, stride, affine: SepConv1d(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x1': lambda C, stride, affine: SepConv1d(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x1': lambda C, stride, affine: SepConv1d(C, C, 7, stride, 3, affine=affine),
    'sep_conv_9x1': lambda C, stride, affine: SepConv1d(C, C, 9, stride, 4, affine=affine),
    'sep_conv_11x1': lambda C, stride, affine: SepConv1d(C, C, 11, stride, 5, affine=affine),

    ##attention layers
    #4 head attention layer
    'multi_attend_4': lambda C, stride, affine: nn.MultiheadAttention(C, 4, batch_first=True),
    'multi_attend_8': lambda C, stride, affine: nn.MultiheadAttention(C, 8, batch_first=True),
    'multi_attend_16': lambda C, stride, affine: nn.MultiheadAttention(C, 16, batch_first=True),
    'ffn': lambda C, stride, affine: FeedForwardNet(C, C*4, 1, stride=1, padding=0, affine=affine),
    'gelu': lambda C, stride, affine: nn.GELU()
    
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class FeedForwardNet(nn.Module):
    def __init__(self, C_in_out, C_hid, kernel_size, stride, padding, affine=True):
        super(FeedForwardNet, self).__init__()
        self.op = nn.Sequential(
            nn.Conv1d(C_in_out, C_hid, kernel_size, stride, bias=False),
            nn.GELU(),
            nn.Conv1d(C_hid, C_in_out, kernel_size, stride=stride, padding=padding, bias=False),
        )
    def forward(self, x):
        return self.op(x)

    
class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)
    
    
class SepConv1d(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv1d, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv1d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)



class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=(padding, 0), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=(padding, 0), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

