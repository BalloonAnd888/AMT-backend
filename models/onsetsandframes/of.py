import torch 
import torch.nn.functional as F 
from torch import nn 

from models.onsetsandframes.lstm import BiLSTM
from preprocessing.mel import MelSpectrogram
from models.utils.constants import DEVICE

class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__() 

        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(in_channels=1, 
                      out_channels=output_features // 16, 
                      kernel_size=(3, 3), 
                      padding=1),
            nn.BatchNorm2d(num_features=output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(in_channels=output_features // 16, 
                      out_channels=output_features // 16, 
                      kernel_size=(3, 3), 
                      padding=1),
            nn.BatchNorm2d(num_features=output_features // 16),
            nn.ReLU(), 
            # layer 2
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=output_features // 16, 
                      out_channels=output_features // 8, 
                      kernel_size=(3, 3), 
                      padding=1),
            nn.BatchNorm2d(num_features=output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=(output_features // 8) * (input_features // 4), 
                      out_features=output_features),
            nn.Dropout(0.5)
        )
    
    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2).flatten(2)
        x = self.fc(x)
        return x 

class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features=input_features,
                      output_features=model_size),
            sequence_model(input_size=model_size, 
                           output_size=model_size),
            nn.Linear(in_features=model_size, 
                      out_features=output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features=input_features,
                      output_features=model_size),
            sequence_model(input_size=model_size, 
                           output_size=model_size),
            nn.Linear(in_features=model_size, 
                      out_features=output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features=input_features,
                      output_features=model_size),   
            nn.Linear(in_features=model_size, 
                      out_features=output_features),
            nn.Sigmoid()         
        )
        self.combined_stack = nn.Sequential(
            sequence_model(input_size=output_features * 3, 
                           output_size=model_size),
            nn.Linear(in_features=model_size, 
                      out_features=output_features),
            nn.Sigmoid()   
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features=input_features,
                      output_features=model_size),   
            nn.Linear(in_features=model_size, 
                      out_features=output_features),            
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred
    
    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        mel_extractor = MelSpectrogram().to(DEVICE)
        mel = mel_extractor(audio_label)

        onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'velocity': velocity_pred.reshape(*velocity_label.shape) 
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses 
    
    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator
