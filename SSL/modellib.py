import torch
import torch.nn as nn
import torch.nn.functional as F


class TriConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttGate(nn.Module):

    def __init__(self, channels, ratio=8):
        super().__init__()
        self.channels = channels
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Linear(self.channels, self.channels // ratio),
            nn.Linear(self.channels // ratio, self.channels)
        )
        self.active = nn.Sigmoid()

    def forward(self, encoder, decoder):
        encoder_feature = self.att(encoder)
        decoder_feature = self.att(decoder)
        att = encoder_feature + decoder_feature
        att = self.active(att)
        output = torch.mul(encoder, att)
        return output


class AttGateQuery(nn.Module):

    def __init__(self, channels, ratio=8, deta=0.5):
        super().__init__()
        self.channels = channels
        self.Pooling = nn.AdaptiveAvgPool3d(1)
        self.Squeeze = nn.Linear(channels, channels // ratio)
        self.Excite = nn.Linear(channels // ratio, channels)
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
        return output

    def _get_SE(self, x):
        se = self.Pooling(x)
        se = se.view(-1, self.channels)
        se = self.Squeeze(se)
        se = self.Excite(se)
        se = se.view(-1, self.channels, 1, 1, 1)
        return se

    def _get_att(self, se_att, queue):
        att = torch.mul(se_att, queue)
        return att

    def _expand_queue_to_5dim(self, queue, se_att):
        expanded_queue = torch.reshape(queue, (se_att.shape[0], queue.shape[0], 1, 1, 1))
        return expanded_queue

    def _updata_queue(self, deta, se_att, queue):
        up_queue = deta * se_att + (1 - deta) * queue
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
    
    
class SEBlock_3D(nn.Module):

    def __init__(self, channels, ratio=8):
        super().__init__()
        self.channels = channels
        self.Pooling = nn.AdaptiveAvgPool3d(1)
        self.Squeeze = nn.Linear(channels, channels // ratio)
        self.Excite = nn.Linear(channels // ratio, channels)

    def forward(self, x):
        se = self.Pooling(x)
        se = se.view(-1, self.channels)
        se = self.Squeeze(se)
        se = self.Excite(se)
        se = se.view(-1, self.channels, 1, 1, 1)
        output = torch.mul(x, se)
        return output


class SEBlock_3D_Queue(nn.Module):

    def __init__(self, channels, ratio=8, deta=0.5):
        super().__init__()
        self.channels = channels
        self.Pooling = nn.AdaptiveAvgPool3d(1)
        self.Squeeze = nn.Linear(channels, channels // ratio)
        self.Excite = nn.Linear(channels // ratio, channels)
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
        return output

    def _get_SE(self, x):
        se = self.Pooling(x)
        se = se.view(-1, self.channels)
        se = self.Squeeze(se)
        se = self.Excite(se)
        se = se.view(-1, self.channels, 1, 1, 1)
        return se

    def _get_att(self, se_att, queue):
        att = torch.mul(se_att, queue)
        return att

    def _expand_queue_to_5dim(self, queue, se_att):
        expanded_queue = torch.reshape(queue, (se_att.shape[0], queue.shape[0], 1, 1, 1))
        return expanded_queue

    def _updata_queue(self, deta, se_att, queue):
        up_queue = deta * se_att + (1 - deta) * queue
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


class SEBlock_3D_Queue_para(nn.Module):

    def __init__(self, channels, ratio=8, deta=0.5):
        super().__init__()
        self.channels = channels
        self.Pooling = nn.AdaptiveAvgPool3d(1)
        self.Squeeze = nn.Linear(channels, channels // ratio)
        self.Excite = nn.Linear(channels // ratio, channels)
        # self.register_buffer('queue', torch.rand(4))
        self.queue = torch.nn.Parameter(torch.rand(4))
        self.deta = deta

    def forward(self, x):
        queue = self._excite_queue()
        se_att = self._get_SE(x)
        queue = self._expand_queue_to_5dim(queue, se_att)
        queue = self._updata_queue(self.deta, se_att, queue)
        output = self._get_att(se_att, queue)
        self.queue = self._squeeze_queue(queue)
        return output

    def _get_SE(self, x):
        se = self.Pooling(x)
        se = se.view(-1, self.channels)
        se = self.Squeeze(se)
        se = self.Excite(se)
        se = se.view(-1, self.channels, 1, 1, 1)
        return se

    def _get_att(self, se_att, queue):
        att = torch.mul(se_att, queue)
        return att

    def _expand_queue_to_5dim(self, queue, se_att):
        expanded_queue = torch.reshape(queue, (se_att.shape[0], queue.shape[0], 1, 1, 1))
        return expanded_queue

    def _updata_queue(self, deta, se_att, queue):
        up_queue = deta * se_att + (1 - deta) * queue
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


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask


class UNet3d_Att_Encoder(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

        self.att1 = SEBlock_3D(n_channels)
        self.att2 = SEBlock_3D(2 * n_channels)
        self.att3 = SEBlock_3D(4 * n_channels)
        self.att4 = SEBlock_3D(8 * n_channels)

    def forward(self, x):
        x1 = self.conv(x)
        # print(x1.shape)
        att1 = self.att1(x1)
        x2 = self.enc1(att1)
        att2 = self.att2(x2)
        x3 = self.enc2(att2)
        att3 = self.att3(x3)
        x4 = self.enc3(att3)
        att4 = self.att4(x4)
        x5 = self.enc4(att4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask


class UNet3d_Att_All(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

        self.att1 = SEBlock_3D(n_channels)
        self.att2 = SEBlock_3D(2 * n_channels)
        self.att3 = SEBlock_3D(4 * n_channels)
        self.att4 = SEBlock_3D(8 * n_channels)

        self.att5 = SEBlock_3D(4 * n_channels)
        self.att6 = SEBlock_3D(2 * n_channels)
        self.att7 = SEBlock_3D(n_channels)
        self.att8 = SEBlock_3D(n_channels)

    def forward(self, x):
        x1 = self.conv(x)
        att1 = self.att1(x1)
        x2 = self.enc1(att1)
        att2 = self.att2(x2)
        x3 = self.enc2(att2)
        att3 = self.att3(x3)
        x4 = self.enc3(att3)
        att4 = self.att4(x4)
        x5 = self.enc4(att4)

        mask = self.dec1(x5, x4)
        att5 = self.att5(mask)
        mask = self.dec2(att5, x3)
        att6 = self.att6(mask)
        mask = self.dec3(att6, x2)
        att7 = self.att7(mask)
        mask = self.dec4(att7, x1)
        att8 = self.att8(mask)
        mask = self.out(att8)
        return mask


class UNet3d_queue_buffer_att(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        att = self.query_att(x1)
        x2 = self.enc1(att)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask

    def excite_att(self, in_att, dim):
        out_att = torch.unsqueeze(in_att, 1)
        out_att = out_att.repeat(1, int(dim)/4)
        out_att = out_att.view(-1)
        return out_att


class UNet3d_queue_para_att(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        att = self.query_att(x1)
        x2 = self.enc1(att)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask

    def excite_att(self, in_att, dim):
        out_att = torch.unsqueeze(in_att, 1)
        out_att = out_att.repeat(1, int(dim)/4)
        out_att = out_att.view(-1)
        return out_att


if __name__ == '__main__':
    input = torch.randn(1, 4, 256, 256, 256)
    output = DoubleConv(4, 24)(input)
    output = SEBlock_3D(channels=24)(output)
    # output = output.detach().numpy()
    # print(output)
