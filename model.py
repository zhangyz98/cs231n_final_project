import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb

# New file added by Yizhi 0521
# New feature: Fine network now has upsampling layers appended; corresponding changes made to: data.py, main.py (see comments in those two for detail changes)

def interleave(tensors, axis):
    '''
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    return tf.reshape(tf.stack(tensors, axis + 1), new_shape)
    '''

    old_shape = list(tensors[0].shape) # (8, 1, 55, 74)
    new_shape = old_shape + [-1] # (8, 1, 55, 74, -1)
    for i in range(len(tensors)):
        tensors[i] = tensors[i].view(new_shape)
    new_shape[axis] *= len(tensors) # (8, 1, 55 * 2, 74, -1)
    return (torch.cat(tensors, dim=axis+1)).view(new_shape[:-1])

'''
def unpool_as_conv(self, size, input_data, id, stride = 1, ReLU = False, BN = True):

	# Model upconvolutions (unpooling + convolution) as interleaving feature
	# maps of four convolutions (A,B,C,D). Building block for up-projections. 


    # Convolution A (3x3)
    # --------------------------------------------------
    layerName = "layer%s_ConvA" % (id)
    self.feed(input_data)
    self.conv( 3, 3, size[3], stride, stride, name = layerName, padding = 'SAME', relu = False)
    outputA = self.get_output()

    # Convolution B (2x3)
    # --------------------------------------------------
    layerName = "layer%s_ConvB" % (id)
    padded_input_B = tf.pad(input_data, [[0, 0], [1, 0], [1, 1], [0, 0]], "CONSTANT")
    self.feed(padded_input_B)
    self.conv(2, 3, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False)
    outputB = self.get_output()

    # Convolution C (3x2)
    # --------------------------------------------------
    layerName = "layer%s_ConvC" % (id)
    padded_input_C = tf.pad(input_data, [[0, 0], [1, 1], [1, 0], [0, 0]], "CONSTANT")
    self.feed(padded_input_C)
    self.conv(3, 2, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False)
    outputC = self.get_output()

    # Convolution D (2x2)
    # --------------------------------------------------
    layerName = "layer%s_ConvD" % (id)
    padded_input_D = tf.pad(input_data, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
    self.feed(padded_input_D)
    self.conv(2, 2, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False)
    outputD = self.get_output()

    # Interleaving elements of the four feature maps
    # --------------------------------------------------
    left = interleave([outputA, outputB], axis=2)  # columns
    right = interleave([outputC, outputD], axis=2)  # columns
    Y = interleave([left, right], axis=3) # rows
        
    if BN:
        layerName = "layer%s_BN" % (id)
        self.feed(Y)
        self.batch_normalization(name = layerName, scale_offset = True, relu = False)
        Y = self.get_output()

    if ReLU:
        Y = tf.nn.relu(Y, name = layerName)
        
    return Y
'''
    

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
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
                
class fineNet(nn.Module):
    def __init__(self, init_weights=True):
        super(fineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 63, kernel_size = 9, stride = 2)
#         self.bn1 = nn.BatchNorm2d(63)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 5, padding = 2)
#         self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, kernel_size = 5, padding = 2)
        # up-sampling layers
        self.conv4_a = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.conv4_b = nn.Conv2d(64, 64, kernel_size = (2, 3))
        self.conv4_c = nn.Conv2d(64, 64, kernel_size = (3, 2))
        self.conv4_d = nn.Conv2d(64, 64, kernel_size = 2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        
        self.pool = nn.MaxPool2d(2)
        if init_weights:
            self._initialize_weights()


    def forward(self, x, y):
                                                # [8, 3, 228, 304]
        x = self.conv1(x)                       # [8, 63, 110, 148]
#         x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)                        # [8, 63, 55, 74]
        x = torch.cat((x,y),1)                  # x - [8, 63, 55, 74] y - [8, 1, 55, 74] => x = [8, 64, 55, 74]
        x = self.conv2(x)                       # [8, 64, 55, 74]
#         x = self.bn2(x)
        x = F.relu(x)
        # x = self.conv3(x)                       # [8, 1, 55, 74]
        # up-sampling layers
        xa = self.conv4_a(x)
        xb = nn.ConstantPad2d((1, 1, 1, 0), 0)(x)
        xb = self.conv4_b(xb)
        xc = nn.ConstantPad2d((1, 0, 1, 1), 0)(x)
        xc = self.conv4_c(xc)
        xd = nn.ConstantPad2d((1, 0, 1, 0), 0)(x)
        xd = self.conv4_d(xd)
        x_left = interleave([xa, xb], 2)
        x_right = interleave([xc, xd], 2)
        x_1 = interleave([x_left, x_right], 3)  # [8, 64, 55*2, 74*2]
        x_2 = x_1.clone()
        x_1 = self.conv5(x_1)
        x = F.relu(x_1 + x_2)
        
        x = self.conv3(x)  # [8, 1, 55*2, 74*2]
        
        return x
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()