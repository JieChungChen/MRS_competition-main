import torch
import torch.nn as nn
from itertools import repeat
from torchsummary import summary


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)


class SpatialDropout(nn.Module):
    """
    spatial dropout是針對channel位置做dropout
    ex.若對(batch, timesteps, embedding)的輸入沿着axis=1執行
    可對embedding的數個channel整體dropout
    沿着axis=2則是對某些timestep整體dropout
    """
    def __init__(self, drop=0.2):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1]) 
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)


def conv_block(in_ch, out_ch, kernel_size, padding, activation=True):
    if activation:
        return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                            nn.Mish())
    else:
        return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size, padding),)


class Predictor(nn.Module): 
    def __init__(self, in_ch, out_ch, drop=0.16):
        super(Predictor, self).__init__()
        self.conv1_1 = conv_block(in_ch, 64, kernel_size=11, padding=5)
        self.conv1_2 = conv_block(64, 128, kernel_size=7, padding=3)
        self.conv1_3 = conv_block(128, 256, kernel_size=5, padding=2)
        self.spacial_drop1 = SpatialDropout(drop)
        self.avgpool1 = nn.AvgPool1d(2, 2)

        # attention layer
        self.conv2_1 = conv_block(256, 64, kernel_size=11, padding=5)
        self.conv2_2 = conv_block(256, 64, kernel_size=7, padding=3)
        self.gloavgpool1 = nn.AvgPool1d(25, 1)
        self.gloavgpool2 = nn.AvgPool1d(25, 1)
        self.conv3 = nn.Sequential(
            conv_block(1, 64, kernel_size=9, padding=4),
            conv_block(64, 32, kernel_size=7, padding=3),
            conv_block(32, 128, kernel_size=7, padding=3)
        )
        self.glomaxpool3 = nn.MaxPool1d(89, 1)
        self.gloavgpool3 = nn.AvgPool1d(89, 1)
        self.linear = nn.Linear(128, out_ch)
    
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.spacial_drop1(x)
        x = self.avgpool1(x)
        att1 = self.conv2_1(x)
        att2 = self.conv2_2(x)
        x = torch.matmul(torch.transpose(att1, 1, 2), att2)
        x = torch.cat((self.gloavgpool1(att1), self.gloavgpool2(x)), dim=1).squeeze().unsqueeze(1)
        conv3 = self.conv3(x)
        conv3 = torch.add(self.glomaxpool3(conv3), self.gloavgpool3(conv3)).squeeze()
        out_rul = self.linear(conv3)
        return out_rul
    