import torch
import torch.nn as nn
import torch.nn.functional as F


class InBlock3d(nn.Module):
    """
    Encoding block used as the first layer in the original 
    Noise2Noise paper (but not Topaz).
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        slope: float=0.1,
    ) -> None:
        """
        Parameters
        ----------
        in_channels: number of input channels
        out_channels: number of filters/output channels
        slope: negative slope used for leaky ReLU
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(slope),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(slope),
            nn.MaxPool3d(2),
        )
        
    def forward(self, x):
        return self.block(x)
    

class EncodingBlock3d(nn.Module):
    """
    3d UNet encoding block.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        slope: float=0.1,
    ) -> None:
        """
        Parameters
        ----------
        in_channels: number of input channels
        out_channels: number of filters/output channels
        slope: negative slope used for leaky ReLU
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(slope),
            nn.MaxPool3d(2),
        )
        
    def forward(self, x):
        return self.block(x)

    
class Bottleneck3d(nn.Module):
    """
    3d UNet bottleneck block.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        slope: float=0.1,
    ) -> None:
        """
        Parameters
        ----------
        in_channels: number of input channels
        out_channels: number of filters/output channels
        slope: negative slope used for leaky ReLU
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(slope),
        )
        
    def forward(self, x):
        return self.block(x)
    
    
class DecodingBlock3d(nn.Module):
    """
    3d UNet decoding block.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        slope: float=0.1,
    ) -> None:
        """
        Parameters
        ----------
        in_channels: number of input channels
        out_channels: number of filters/output channels
        slope: negative slope used for leaky ReLU
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(slope),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(slope),
        )

    def forward(self, x, skip_residual):
        """
        In addition to applying layer operations, add information 
        from the corresponding encoding layer (skip connection).
        
        To-do: test F.interpolate with
        mode='bilinear', align_corners=True
        """
        up_x = F.interpolate(x, size=tuple(skip_residual.size()[2:]), mode='nearest')
        concat_x = torch.cat([up_x, skip_residual], dim=1)
        return self.block(concat_x)
    
    
class OutBlock3d(nn.Module):
    """
    3d UNet final decoding block.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        slope: float=0.1,
    ) -> None:
        """
        Parameters
        ----------
        in_channels: number of input channels
        mid_channels: intermediate number of channels
        out_channels: number of filters/output channels
        slope: negative slope used for leaky ReLU
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            nn.LeakyReLU(slope),
            nn.Conv3d(mid_channels, int(mid_channels/2), 3, padding=1),
            nn.LeakyReLU(slope),
            nn.Conv3d(int(mid_channels/2), out_channels, 3, padding=1),
        )

    def forward(self, x, skip_residual):
        """
        In addition to applying layer operations, add information 
        from the corresponding encoding layer (skip connection).
        
        To-do: test F.interpolate with
        mode='bilinear', align_corners=True
        """
        up_x = F.interpolate(x, size=tuple(skip_residual.size()[2:]), mode='nearest')
        concat_x = torch.cat([up_x, skip_residual], dim=1)
        return self.block(concat_x)


class UNet3d(nn.Module):
    """
    3D UNet architecture similar to Topaz's architecture as described 
    in Bepler, Kelley, Noble, and Berger, Nature Communications, 2020.
    """
    def __init__(self, n_filters: int=48, slope: float=0.1):
        """ Initialize a 3D U-Net. """
        super().__init__()
        
        self.encoding1 = EncodingBlock3d(1, n_filters, slope=slope)
        self.encoding2 = EncodingBlock3d(n_filters, n_filters, slope=slope)
        self.encoding3 = EncodingBlock3d(n_filters, n_filters, slope=slope)
        self.encoding4 = EncodingBlock3d(n_filters, n_filters, slope=slope)
        self.encoding5 = EncodingBlock3d(n_filters, n_filters, slope=slope)
        self.bottleneck = Bottleneck3d(n_filters, n_filters, slope=slope)
        self.decoding5 = DecodingBlock3d(2*n_filters, 2*n_filters, slope=slope)
        self.decoding4 = DecodingBlock3d(3*n_filters, 2*n_filters, slope=slope)
        self.decoding3 = DecodingBlock3d(3*n_filters, 2*n_filters, slope=slope)
        self.decoding2 = DecodingBlock3d(3*n_filters, 2*n_filters, slope=slope)
        self.decoding1 = OutBlock3d(2*n_filters+1, int(4./3*n_filters), 1, slope=slope)

    def count_parameters(self):
        """ Count the number of model paramters. """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Pass data through encoder, bottleneck, and decoder layers 
        with skip connections.
        """
        x1 = self.encoding1(x)
        x2 = self.encoding2(x1)
        x3 = self.encoding3(x2)
        x4 = self.encoding4(x3)
        x5 = self.encoding5(x4)
        y = self.bottleneck(x5)
        y = self.decoding5(y, x4)
        y = self.decoding4(y, x3)
        y = self.decoding3(y, x2)
        y = self.decoding2(y, x1)
        y = self.decoding1(y, x)
        
        return y
