import torch
from torch import nn
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.image as mpimg
import numpy as np



class KerasConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, **kwargs) -> None:
      super().__init__()

      self.conv = nn.Conv2d(
          in_channels=in_channels, out_channels=out_channels,
          kernel_size=kernel_size, **kwargs
      )
      self.activation = nn.ReLU()

  def forward(self, x):
    x = self.conv(x)
    x = self.activation(x)

    return x

class KerasUp(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, **kwargs) -> None:
      super().__init__()

      self.upsample = nn.Upsample(scale_factor=(2,2))
      self.conv1 = KerasConv(in_channels, out_channels, kernel_size=(2,2), **kwargs)
      self.activation = nn.ReLU()

      self.conv2 = KerasConv(out_channels*2, out_channels, kernel_size, **kwargs)
      self.conv3 = KerasConv(out_channels, out_channels, kernel_size, **kwargs)

  def forward(self, x, c):
    x = self.upsample(x)
    x = self.conv1(x)
    x = self.activation(x)

    x = torch.cat((c,x), dim=1)

    x = self.conv2(x)
    x = self.conv3(x)

    return x

class SegNet(nn.Module):
  def __init__(self, kernel_size=(3,3)) -> None:
      super().__init__()

      self.conv1_1 = KerasConv(in_channels=1, out_channels=64, kernel_size=kernel_size, padding="same")
      self.conv1_2 = KerasConv(in_channels=64, out_channels=64, kernel_size=kernel_size, padding="same")
      self.pool1 = nn.MaxPool2d((2,2))
      self.layer_1 = nn.Sequential(
          self.conv1_1, self.conv1_2
      )

      self.conv2_1 = KerasConv(in_channels=64, out_channels=128, kernel_size=kernel_size, padding="same")
      self.conv2_2 = KerasConv(in_channels=128, out_channels=128, kernel_size=kernel_size, padding="same")
      self.pool2 = nn.MaxPool2d((2,2))
      self.layer_2 = nn.Sequential(
          self.conv2_1, self.conv2_2
      )

      self.conv3_1 = KerasConv(in_channels=128, out_channels=256, kernel_size=kernel_size, padding="same")
      self.conv3_2 = KerasConv(in_channels=256, out_channels=256, kernel_size=kernel_size, padding="same")
      self.pool3 = nn.MaxPool2d((2,2))
      self.layer_3 = nn.Sequential(
          self.conv3_1, self.conv3_2
      )

      self.conv4_1 = KerasConv(in_channels=256, out_channels=512, kernel_size=kernel_size, padding="same")
      self.conv4_2 = KerasConv(in_channels=512, out_channels=512, kernel_size=kernel_size, padding="same")
      self.drop4 = nn.Dropout(0.5)
      self.pool4 = nn.MaxPool2d((2,2))
      self.layer_4 = nn.Sequential(
          self.conv4_1, self.conv4_2, self.drop4
      )

      self.conv5_1 = KerasConv(in_channels=512, out_channels=1024, kernel_size=kernel_size, padding="same")
      self.conv5_2 = KerasConv(in_channels=1024, out_channels=1024, kernel_size=kernel_size, padding="same")
      self.drop5 = nn.Dropout(0.5)
      self.layer_5 = nn.Sequential(
          self.conv5_1, self.conv5_2, self.drop5
      )

      self.layer_6 = KerasUp(1024, 512, kernel_size, padding="same")
      self.layer_7 = KerasUp(512, 256, kernel_size, padding="same")
      self.layer_8 = KerasUp(256, 128, kernel_size, padding="same")
      self.layer_9 = KerasUp(128, 64, kernel_size, padding="same")

      self.layer_10 = nn.Sequential(
          KerasConv(in_channels=64, out_channels=2, kernel_size=kernel_size, padding="same"),
          nn.Conv2d(2, 1, (1,1), padding="same"),
      )

  def forward(self, x):
    layer_1 = self.layer_1(x)
    layer_2 = self.layer_2(self.pool1(layer_1))
    layer_3 = self.layer_3(self.pool2(layer_2))
    layer_4 = self.layer_4(self.pool3(layer_3))
    layer_5 = self.layer_5(self.pool4(layer_4))
    layer_6 = self.layer_6(layer_5, layer_4)
    layer_7 = self.layer_7(layer_6, layer_3)
    layer_8 = self.layer_8(layer_7, layer_2)
    layer_9 = self.layer_9(layer_8, layer_1)
    layer_10 = self.layer_10(layer_9)

    return layer_10

