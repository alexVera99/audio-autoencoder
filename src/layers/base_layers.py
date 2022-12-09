import torch.nn as nn


# Convolution + BatchNormnalization + ReLU or Tanh block for the encoder
class ConvBNReLU(nn.Module):
  def __init__(self,in_channels, out_channels, activation: str = "relu"):
    super(ConvBNReLU, self).__init__()
    self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,
                          padding = 1)
    self.bn = nn.BatchNorm2d(out_channels)

    if activation.lower() == "relu":
      self.activation = nn.ReLU(inplace=True)
    elif activation.lower() == "tanh":
      self.activation = nn.Tanh()
    else:
      raise Exception("Activations available are: relu or tanh")

  def forward(self,x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.activation(out)   
    return out


# Downsampling Convolution + BatchNormnalization + ReLU block for the encoder
class DownsamplinConvBNReLU(nn.Module):
  def __init__(self,in_channels, out_channels):
    super(DownsamplinConvBNReLU, self).__init__()
    self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,
                          stride= 2, padding = 1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self,x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.relu(out)   
    return out

# Upsampling Convolution (Deconvolution) + 
# BatchNormnalization + ReLU block for the encoder
class UpsamplinConvBNReLU(nn.Module):
  def __init__(self,in_channels, out_channels, output_padding: list):
    super(UpsamplinConvBNReLU, self).__init__()
    self.deconv = nn.ConvTranspose2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride= 2, 
                                     padding = 1, 
                                     output_padding = output_padding)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self,x):
    out = self.deconv(x)
    out = self.bn(out)
    out = self.relu(out)   
    return out