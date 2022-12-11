import torch
from src.layers.base_layers import *

class Encoder(nn.Module):
  def __init__(self, input_channels = 1, base_channels=32, out_features = 400):
    super(Encoder, self).__init__()

    self.layer1 = ConvBNReLU(input_channels, 
                             base_channels)
    
    self.layer2 = DownsamplinConvBNReLU(base_channels, 
                                        base_channels*2)
    
    self.layer3 = DownsamplinConvBNReLU(base_channels*2,
                                        base_channels*4)
    
    self.layer4 = DownsamplinConvBNReLU(base_channels*4,
                                        base_channels*8)
    
    self.layer5 = DownsamplinConvBNReLU(base_channels*8,
                                        base_channels*16)
    
    self.layer6 = DownsamplinConvBNReLU(base_channels*16,
                                        base_channels*32)

    fc_input_size = 14 * 17 * base_channels*32 # Hardcoded for input of (431 x 513)
    self.fc = nn.Linear(fc_input_size, out_features)
  
  def forward(self,x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)

    return self.fc(out.view(x.shape[0],-1))

class Decoder(nn.Module):
  def __init__(self, out_channels = 1, 
               base_channels=32,
               input_features = 400):
    super(Decoder, self).__init__()

    fc_input_size = 14 * 17 * base_channels*32 # Hardcoded for input of (431 x 513)
    self.fc = nn.Linear(input_features, 
                        fc_input_size)

    self.layer6 = UpsamplinConvBNReLU(base_channels*32,
                                      base_channels*16,
                                      [0, 0])

    self.layer5 = UpsamplinConvBNReLU(base_channels*16,
                                      base_channels*8,
                                      [0, 1])

    self.layer4 = UpsamplinConvBNReLU(base_channels*8,
                                      base_channels*4,
                                      [0, 1])

    self.layer3 = UpsamplinConvBNReLU(base_channels*4,
                                      base_channels*2,
                                      [0, 1])

    self.layer2 = UpsamplinConvBNReLU(base_channels*2,
                                      base_channels,
                                      [0, 0])

    self.layer1 = ConvBNReLU(base_channels, 
                             out_channels, 
                             activation = "tanh")
    
    self.base_channels = base_channels
    
  
  def forward(self,x):
    out = self.fc(x)
    out = out.view(x.shape[0],
                   self.base_channels*32,
                   17,
                   14) # Hardcoded for input of (431 x 513)
    out = self.layer6(out)
    out = self.layer5(out)
    out = self.layer4(out)
    out = self.layer3(out)
    out = self.layer2(out)
    out = self.layer1(out)
    return out

class AudioAutoEncoder(nn.Module):
    def __init__(self, input_channels, base_channels, num_features):
        super(AudioAutoEncoder, self).__init__()
        # Initialize the encoder and decoder using a dimensionality out_features for the vector z
        self.encoder = Encoder(input_channels, base_channels, num_features)
        self.decoder = Decoder(input_channels, base_channels, num_features)


    def encode(self,x):
        return self.encoder(x)

    def forward(self,x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode