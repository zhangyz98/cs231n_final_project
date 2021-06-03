import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torchvision.models as models


class coarseNet(nn.Module):
    def __init__(self,init_weights=True):
        super(coarseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0)
        # added batchnorm(1-5) by Yizhi (0511)
#         self.batchnorm1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, padding = 2)
#         self.batchnorm2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, padding = 1)
#         self.batchnorm3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, padding = 1)
#         self.batchnorm4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 2)
#         self.batchnorm5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(12288, 4096)
        self.fc2 = nn.Linear(4096, 4070)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d()
        if init_weights:
            self._initialize_weights()


    def forward(self, x):
                                                # [n, c,  H,   W ]
                                                # [8, 3, 228, 304]
        x = self.conv1(x)                       # [8, 96, 55, 74]
#         x = self.batchnorm1(x) # Yizhi
        x = F.relu(x)
        x = self.pool(x)                        # [8, 96, 27, 37] -- 
        x = self.conv2(x)                       # [8, 256, 23, 33]
#         x = self.batchnorm2(x) # Yizhi
        x = F.relu(x)
        x = self.pool(x)                        # [8, 256, 11, 16] 18X13
        x = self.conv3(x)                       # [8, 384, 9, 14]
#         x = self.batchnorm3(x) # Yizhi
        x = F.relu(x)
        x = self.conv4(x)                       # [8, 384, 7, 12]
#         x = self.batchnorm4(x) # Yizhi
        x = F.relu(x)
        x = self.conv5(x)                       # [8, 256, 5, 10] 8X5
#         x = self.batchnorm5(x) # Yizhi
        x = F.relu(x)
        x = x.view(x.size(0), -1)               # [8, 12800]
        x = F.relu(self.fc1(x))                 # [8, 4096]
        x = self.dropout(x)
        x = self.fc2(x)                         # [8, 4070]     => 55x74 = 4070
        x = x.view(-1, 1, 55, 74)
        return x


    # Pre-train Imagenet Model ??
    # Why random guassian model.    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
#                 nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
                
                
class fineNet(nn.Module):
    def __init__(self, init_weights=True):
        super(fineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 63, kernel_size = 9, stride = 2)
#         self.bn1 = nn.BatchNorm2d(63)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 5, padding = 2)
#         self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, kernel_size = 5, padding = 2)
        self.resnet = ResnetBlock(3, 'zero', nn.BatchNorm2d, 1, 0)
        self.pool = nn.MaxPool2d(2)
        if init_weights:
            self._initialize_weights()


    def forward(self, x, y):
#         print(x.shape)
        x = self.resnet(x)
#         print(x.shape)
                                                # [8, 3, 228, 304]
        x = self.conv1(x)                       # [8, 63, 110, 148]
#         x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)                        # [8, 63, 55, 74]
        x = torch.cat((x,y),1)                  # x - [8, 63, 55, 74] y - [8, 1, 55, 74] => x = [8, 64, 55, 74]
        x = self.conv2(x)                       # [8, 64, 55, 74]
#         x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)                       # [8, 64, 55, 74]
        return x
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
#                 m.weight.data.normal_(0, 0.01)
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
        
        
        
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.2)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

