import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, channels=1, num_layers=17):
      super(DnCNN, self).__init__()

      # first layer  (conv + ReLU)
      layers = [nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True)]

      # intermediate layers (conv + batchnorm + ReLU)
      for _ in range(num_layers-2):
        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))

      # last layer (conv only)
      layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1, bias=False))

      self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x) # predict the noise
        return x-noise # denoised image = noisy image - predicted noise