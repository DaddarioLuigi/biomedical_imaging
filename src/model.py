from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Computes Dice coefficient for binary masks.
Parameters:
    y_true: (N,1,D,H,W) float tensor in {0,1}
    y_pred: (N,1,D,H,W) float tensor in [0,1]
"""
def dice_coefficient_torch(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    y_true = y_true.float()
    y_pred = y_pred.float()
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)


def dice_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return 1.0 - dice_coefficient_torch(y_true, y_pred)


def iou_coefficient_torch(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    y_true = y_true.float()
    y_pred = y_pred.float()
    intersection = torch.sum(y_true * y_pred)
    total = torch.sum(y_true) + torch.sum(y_pred)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

"""
This is a small building block for 3D CNNs.
It applies a 3D convolution, then batch normalization, then a ReLU activation.
Use it to extract features while keeping code concise and consistent.
"""
class Conv3DBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reg_l2: float = 0.0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.reg_l2 = reg_l2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


"""
This is an encoder block that downsamples features.
It runs two Conv3D+BN+ReLU layers, keeps a skip copy, then downsamples.
A residual 1x1 path matches shape so we can add it to the downsampled main path.
"""
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reg_l2: float = 0.0, p_dropout: float = 0.0):
        super().__init__()
        self.conv1 = Conv3DBNReLU(in_channels, out_channels, reg_l2)
        self.conv2 = Conv3DBNReLU(out_channels, out_channels, reg_l2)
        self.dropout = nn.Dropout3d(p=p_dropout) if p_dropout > 0 else nn.Identity()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.resid = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False)

    def forward(self, x: torch.Tensor):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.dropout(x1)
        skip = x1
        main = self.pool(x1)
        resid = self.resid(x)
        out = main + resid
        return out, skip

"""
This is a decoder block that upsamples features.
It upsamples the input, concatenates the matching encoder skip, then refines with convs.
A residual upsampled path is also added for stable learning and better gradients.
"""
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, reg_l2: float = 0.0,
                 use_transpose: bool = False, p_dropout: float = 0.0):
        super().__init__()
        if use_transpose:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            up_out_channels = out_channels
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
            )
            up_out_channels = out_channels

        self.conv1 = Conv3DBNReLU(up_out_channels + skip_channels, out_channels, reg_l2)
        self.conv2 = Conv3DBNReLU(out_channels, out_channels, reg_l2)
        self.dropout = nn.Dropout3d(p=p_dropout) if p_dropout > 0 else nn.Identity()
        self.resid_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        x2 = torch.cat([up, skip], dim=1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.dropout(x2)
        resid = self.resid_up(x)
        out = x2 + resid
        return out


"""
This class builds the full 3D U-Net with residual connections.
It encodes the input volume, processes a bottleneck, then decodes with skip connections.
The output is a single-channel probability map (sigmoid) for segmentation.
"""
class UNet3DResidual(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 16,
        reg_l2: float = 1e-5,
        p_dropout_enc: float = 0.1,
        p_dropout_dec: float = 0.1,
        p_dropout_bot: float = 0.1,
        use_transpose: bool = False,
    ):
        super().__init__()
        bf = base_filters
        self.down1 = DownBlock(in_channels, bf, reg_l2, p_dropout_enc)
        self.down2 = DownBlock(bf, bf * 2, reg_l2, p_dropout_enc)
        self.down3 = DownBlock(bf * 2, bf * 4, reg_l2, p_dropout_enc)
        self.down4 = DownBlock(bf * 4, bf * 8, reg_l2, p_dropout_enc)

        self.bot1 = Conv3DBNReLU(bf * 8, bf * 16, reg_l2)
        self.bot_drop = nn.Dropout3d(p=p_dropout_bot) if p_dropout_bot > 0 else nn.Identity()
        self.bot2 = Conv3DBNReLU(bf * 16, bf * 16, reg_l2)

        self.up3 = UpBlock(bf * 16, bf * 8, bf * 8, reg_l2, use_transpose, p_dropout_dec)
        self.up2 = UpBlock(bf * 8, bf * 4, bf * 4, reg_l2, use_transpose, p_dropout_dec)
        self.up1 = UpBlock(bf * 4, bf * 2, bf * 2, reg_l2, use_transpose, p_dropout_dec)
        self.up0 = UpBlock(bf * 2, bf, bf, reg_l2, use_transpose, p_dropout_dec)

        self.out_conv = nn.Conv3d(bf, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,1,D,H,W)
        x1, s1 = self.down1(x)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4, s4 = self.down4(x3)

        b = self.bot1(x4)
        b = self.bot_drop(b)
        b = self.bot2(b)

        u3 = self.up3(b, s4)
        u2 = self.up2(u3, s3)
        u1 = self.up1(u2, s2)
        u0 = self.up0(u1, s1)

        out = self.out_conv(u0)
        out = torch.sigmoid(out)
        return out


# ---------------------------
# BUILD MODEL
#Returns a PyTorch UNet3DResidual
# ---------------------------
def build_unet_3d(
    input_shape: Tuple[int, int, int, int] = (64, 64, 64, 1),
    base_filters: int = 16,
    reg_l2: float = 1e-5,
    p_dropout_enc: float = 0.1,
    p_dropout_dec: float = 0.1,
    p_dropout_bot: float = 0.1,
    use_transpose: bool = False
) -> nn.Module:
    """
   
    """
    model = UNet3DResidual(
        in_channels=1,
        base_filters=base_filters,
        reg_l2=reg_l2,
        p_dropout_enc=p_dropout_enc,
        p_dropout_dec=p_dropout_dec,
        p_dropout_bot=p_dropout_bot,
        use_transpose=use_transpose,
    )
    return model


