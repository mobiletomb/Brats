import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import attention

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_groups=2):
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


class AttUp(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)
        self.att = attention.AttGate(in_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.att(x)
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


class AttUNet3d(nn.Module):
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

        self.dec1 = AttUp(16 * n_channels, 4 * n_channels)
        self.dec2 = AttUp(8 * n_channels, 2 * n_channels)
        self.dec3 = AttUp(4 * n_channels, n_channels)
        self.dec4 = AttUp(2 * n_channels, n_channels)
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


class Double_Path_UNet3D(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels, get_pair_feature=False, gamma=0.5):
        super().__init__()
        self.n_classes = n_classes
        self.get_pair_feature = get_pair_feature
        self.paired_channels = in_channels // 2
        self.paired_nchannels = n_channels // 2
        self.gamma = gamma
        self.mid_feature = []
        self.t1_feature = []
        self.t2_feature = []

        self.conv_t1 = DoubleConv(self.paired_channels, self.paired_nchannels)
        self.enc1_t1 = Down(self.paired_nchannels, 2 * self.paired_nchannels)
        self.enc2_t1 = Down(2 * self.paired_nchannels, 4 * self.paired_nchannels)
        self.enc3_t1 = Down(4 * self.paired_nchannels, 8 * self.paired_nchannels)
        self.enc4_t1 = Down(8 * self.paired_nchannels, 8 * self.paired_nchannels)

        self.conv_t2 = DoubleConv(self.paired_channels, self.paired_nchannels)
        self.enc1_t2 = Down(self.paired_nchannels, 2 * self.paired_nchannels)
        self.enc2_t2 = Down(2 * self.paired_nchannels, 4 * self.paired_nchannels)
        self.enc3_t2 = Down(4 * self.paired_nchannels, 8 * self.paired_nchannels)
        self.enc4_t2 = Down(8 * self.paired_nchannels, 8 * self.paired_nchannels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, t1_Pair, t2_Pair):
        t1_en1 = self.conv_t1(t1_Pair)
        t1_en2 = self.enc1_t1(t1_en1)
        t1_en3 = self.enc2_t1(t1_en2)
        t1_en4 = self.enc3_t1(t1_en3)
        t1_en5 = self.enc4_t1(t1_en4)

        # print("t1_en1.shape:", t1_en1.shape)
        # print("t1_en2.shape:", t1_en2.shape)
        # print("t1_en3.shape:", t1_en3.shape)
        # print("t1_en4.shape:", t1_en4.shape)
        # print("t1_en5.shape:", t1_en5.shape)

        t2_en1 = self.conv_t2(t2_Pair)
        t2_en2 = self.enc1_t2(t2_en1)
        t2_en3 = self.enc2_t2(t2_en2)
        t2_en4 = self.enc3_t2(t2_en3)
        t2_en5 = self.enc4_t2(t2_en4)

        mid_2 = self._get_EMA(t1_en2, t2_en2, self.gamma)
        mid_3 = self._get_EMA(t1_en3, t2_en3, self.gamma)
        mid_4 = self._get_EMA(t1_en4, t2_en4, self.gamma)

        en1 = torch.cat((t1_en1, t2_en1), dim=1)
        en2 = torch.cat((t1_en2, t2_en2), dim=1)
        en3 = torch.cat((t1_en3, t2_en3), dim=1)
        en4 = torch.cat((t1_en4, t2_en4), dim=1)
        en5 = torch.cat((t1_en5, t2_en5), dim=1)

        mask = self.dec1(en5, en4)
        # print("mask.shape:", mask.shape)
        mask = self.dec2(mask, en3)
        # print("mask.shape:", mask.shape)
        mask = self.dec3(mask, en2)
        # print("mask.shape:", mask.shape)
        mask = self.dec4(mask, en1)
        # print("mask.shape:", mask.shape)

        out = self.out(mask)

        if self.get_pair_feature:
            for i in range(3):
                self.mid_feature.append(f"mid_{i + 2}")
                self.t1_feature.append(f"t1_en{i + 2}")
                self.t2_feature.append(f"t2_en{i + 2}")
            return out, self.mid_feature, self.t1_feature, self.t2_feature
        else:
            return out

    def _get_EMA(self, x, y, gama):
        return x + torch.mul(y, gama)


if __name__ == '__main__':
    inps = torch.arange(1 * 4 * 240 * 240 * 155, dtype=torch.float32).view(1, 4, 240, 240, 155)
    tgts = torch.arange(1 * 4 * 240 * 240 * 155, dtype=torch.float32).view(1, 4, 240, 240, 155)
    dataset = TensorDataset(inps, tgts)
    loader = DataLoader(dataset,
                        batch_size=1,
                        pin_memory=True,
                        num_workers=2)

    device = 'cpu'
    model = Double_Path_UNet3D(in_channels=4, n_classes=3, n_channels=24)
    model.to(device)
    for data, target in loader:
        data = data.to(device)
        t1 = data[:, :2, :, :, :]
        t2 = data[:, 2:4, :, :, :]
        output = model(t1, t2)
        print(output.shape)


