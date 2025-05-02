import torch
from torchvision import models
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class DrivingPlanner(nn.Module):
    def __init__(self, future_steps=60, history_steps=21, history_encoder=None, step_dim=3, hidden_dim=128, ego_encoding=False):
        super().__init__()

        print(f"Using history encoding: {history_encoder} and ego: {ego_encoding}")

        # CNN for processing camera images
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])  
        cnn_out_dim = base_model.fc.in_features

        # RNN for processing history
        if history_encoder == None or history_encoder == 'None':
            self.rnn = None
            rnn_out_dim = history_steps * step_dim
        elif history_encoder == 'GRU':
            self.rnn = nn.GRU(input_size=3, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
            rnn_out_dim = hidden_dim
        elif history_encoder == 'LSTM':
            self.rnn = nn.LSTM(input_size=3, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
            rnn_out_dim = hidden_dim * 2
        else:
            history_encoder = None
            self.rnn = None
            rnn_out_dim = history_steps * step_dim

        self.history_encoder = history_encoder

        if ego_encoding:
            self.ego_encoder = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            ego_out_dim = 32
        else:
            self.ego_encoder = None
            ego_out_dim = 3

        # Decoder for predicting future trajectory
        self.decoder = nn.Sequential(
            nn.Linear(cnn_out_dim + rnn_out_dim + ego_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, future_steps * step_dim)
        )

    def forward(self, camera, history):

        # Process camera images
        visual_features = self.cnn(camera).view(camera.size(0), -1)

        # Process history
        if self.rnn == None:
            history_feat = history.reshape(history.size(0), -1)
        elif self.history_encoder == "GRU":
            _, history_last = self.rnn(history)
            history_feat = history_last[-1]
        else:
            _, (h_n, c_n) = self.rnn(history)
            # Concat the long term and short term memory
            history_feat = torch.cat((h_n[-1], c_n[-1]), dim=1)

        # Process ego
        ego = history[:,-2]
        if self.ego_encoder:
            ego_feat = self.ego_encoder(ego)
        else:
            ego_feat = ego

        # Combine features
        combined = torch.cat([visual_features, history_feat, ego_feat], dim=1)

        # Predict future trajectory
        future = self.decoder(combined)
        future = future.reshape(-1, 60, 3)  # Reshape to (batch_size, timesteps, features)

        return future