import torch
import torch.nn as nn
import torch.nn.functional as F

import math

#########################  SE #########################
class SEGate(nn.Module):

    def __init__(self, channels, ratio=8):
        super().__init__()
        self.channels = channels
        self.pooling = nn.AdaptiveAvgPool3d(1)
        self.att = nn.Sequential(
            nn.Linear(self.channels, self.channels // ratio),
            nn.ReLU(),
            nn.Linear(self.channels // ratio, self.channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.pooling(x)
        att = att.view(-1, self.channels)
        att = self.att(att)
        att = att.view(-1, self.channels, 1, 1, 1)
        output = torch.mul(x, att)
        return output


class SEQueue(nn.Module):

    def __init__(self, channels, ratio=8, deta=0.900):
        super().__init__()
        self.channels = channels
        self.Pooling = nn.AdaptiveAvgPool3d(1)
        self.att = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid()
        )

        self.register_buffer('queue', torch.rand(4))
        # self.queue = torch.nn.Parameter(torch.rand(4))

        self.deta = deta

    def forward(self, x):
        queue = self._excite_queue()
        se_att = self._get_SE(x)
        queue = self._expand_queue_to_5dim(queue, se_att)
        queue = self._updata_queue(self.deta, se_att, queue)
        output = self._get_att(se_att, queue)
        self.queue = self._squeeze_queue(queue)
        output = torch.mul(x, output)
        return output

    def _get_SE(self, x):
        se = self.Pooling(x)
        se = se.view(-1, self.channels)
        se = torch.squeeze(se, 0)
        se = self.att(se)
        se = se.view(-1, self.channels, 1, 1, 1)
        return se

    def _get_att(self, se_att, queue):
        att = torch.mul(se_att, queue)
        return att

    def _expand_queue_to_5dim(self, queue, se_att):
        B = se_att.shape[0]
        expanded_queue = torch.reshape(queue, (1, queue.shape[0], 1, 1, 1))
        expanded_queue = expanded_queue.repeat(B, 1, 1, 1, 1)
        return expanded_queue

    def _updata_queue(self, deta, se_att, queue):
        up_queue = (1 - deta) * se_att.detach().clone() + deta * queue
        return up_queue

    def _excite_queue(self):
        excited_queue = torch.unsqueeze(self.queue, 1)
        excited_queue = excited_queue.repeat(1, int(self.channels / 4))
        excited_queue = excited_queue.view(-1)
        return excited_queue

    def _squeeze_queue(self, queue):
        queue = torch.reshape(queue, (4, -1))
        queue = torch.mean(queue, 1)
        return queue

#########################  SE #########################


######################### CBAM #########################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=None, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        B, C, _, _, _ = x.shape
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum.reshape(B, C, 1, 1, 1))
        return x * scale


class ChannelGate_Queue(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate_Queue, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = SEQueue(gate_channels, ratio=8, deta=0.900)
        self.pool_types = pool_types

    def forward(self, x):
        B, C, _, _, _ = x.shape
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum.reshape(B, C, 1, 1, 1))
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class CBAMQueue(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAMQueue, self).__init__()
        self.ChannelGate = ChannelGate_Queue(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

######################### CBAM #########################


######################### BAM  #########################
# class ChannelGate(nn.Module):
#     def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
#         super(ChannelGate, self).__init__()
#         self.gate_activation = gate_activation
#         self.gate_c = nn.Sequential()
#         self.gate_c.add_module( 'flatten', Flatten() )
#         gate_channels = [gate_channel]
#         gate_channels += [gate_channel // reduction_ratio] * num_layers
#         gate_channels += [gate_channel]
#         for i in range( len(gate_channels) - 2 ):
#             self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
#             self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
#             self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
#         self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
#     def forward(self, in_tensor):
#         avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
#         return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)
#
#
# class SpatialGate(nn.Module):
#     def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
#         super(SpatialGate, self).__init__()
#         self.gate_s = nn.Sequential()
#         self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
#         self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
#         self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
#         for i in range( dilation_conv_num ):
#             self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
# 						padding=dilation_val, dilation=dilation_val) )
#             self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
#             self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
#         self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
#     def forward(self, in_tensor):
#         return self.gate_s( in_tensor ).expand_as(in_tensor)
#
#
# class BAM(nn.Module):
#     def __init__(self, gate_channel):
#         super(BAM, self).__init__()
#         self.channel_att = ChannelGate(gate_channel)
#         self.spatial_att = SpatialGate(gate_channel)
#     def forward(self,in_tensor):
#         att = 1 + torch.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
#         return att * in_tensor
######################### CBAM #########################

# to do
# 3d cbam queue
# 2d cbam queue  done
# stem block

if __name__ == "__main__":
    inp = torch.arange(2 * 256 * 8 * 8 * 8, dtype=torch.float32).view(2, 256, 8, 8, 8)
    act = SEGate(256, ratio=8)
    out = act(inp)
    print(out.shape)