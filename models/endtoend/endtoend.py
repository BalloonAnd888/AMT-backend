import torch 
from torch import nn 

class ETE(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()

        self.conv_layers = nn.Sequential(
            self.conv_block(in_chan=input_shape, out_chan=32),
            self.conv_block(in_chan=32, out_chan=64),
            self.conv_block(in_chan=64, out_chan=128),
            self.conv_block(in_chan=128, out_chan=256),
            self.conv_block(in_chan=256, out_chan=512)
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 20, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_shape)
        )
    
    def conv_block(self, in_chan, out_chan):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, 
                      out_channels=out_chan, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_chan),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
