
import torch
import torch.nn as nn
import torchvision.models as models

class DepthEncoder(nn.Module):
    def __init__(self, input_dim=1280):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 512, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=5, stride=2, padding=2),
            nn.Sigmoid(),
            nn.Upsample(size=(200,300), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        depth = self.encoder(x)
        return depth
    
class SemanticEncoder(nn.Module):
    def __init__(self, input_dim=1280, num_channels=15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 512, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, num_channels, kernel_size=5, stride=2, padding=2),
            nn.Upsample(size=(200,300), mode='bilinear', align_corners=False)
        )


    def forward(self, x):
        # x: (B, 1, H, W)
        depth = self.encoder(x)
        return depth


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        baseModel = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(baseModel.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, 512)

    def forward(self, x):
        # x: (B, 3, H, W)
        feat = self.backbone(x)
        pool = self.avg_pool(feat)  
        return self.fc(pool.view(x.size(0), -1)), feat  # (B, 512)

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x: (B, 20, 2) = 20 past positions (2D)
        _, h = self.gru(x)
        return h[-1]

class TrajectoryDecoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=3, horizon=60):
        super().__init__()
        self.gru = nn.GRU(input_dim, 128, batch_first=True)
        self.output_layer = nn.Linear(128, output_dim)
        self.horizon = horizon

    def forward(self, x):
        # x: (B, feat_dim)
        x = x.unsqueeze(1).repeat(1, self.horizon, 1)  # Repeat for decoder
        out, _ = self.gru(x)
        out = self.output_layer(out)  # (B, T, 2)
        return out

class FusionModule(nn.Module):
    def __init__(self, rgb_dim=512, pos_dim=128, out_dim=256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(rgb_dim + pos_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, rgb_feat, pos_feat):
        x = torch.cat([rgb_feat, pos_feat], dim=1)
        return self.fuse(x)  # (B, out_dim)


        return self.output_layer(out)  # (B, 60, 2)

def smooth_trajectory(path, kernel_size=5, gaussian=False):
    """
    Smooths the predicted trajectory (B, T, 2) using 1D convolution.
    """
    B, T, D = path.shape
    path = path.transpose(1, 2)  # (B, 2, T)

    if gaussian:
        # Create Gaussian kernel
        base = torch.arange(kernel_size).float() - kernel_size // 2
        kernel = torch.exp(-0.5 * (base / (kernel_size / 2))**2)
    else:
        # Uniform kernel
        kernel = torch.ones(kernel_size)

    kernel /= kernel.sum()  # normalize
    kernel = kernel.view(1, 1, -1).to(path.device)  # (1, 1, K)
    kernel = kernel.repeat(D, 1, 1)  # (2, 1, K)

    # Apply 1D conv
    smoothed = F.conv1d(path, kernel, padding=kernel_size // 2, groups=D)
    smoothed = smoothed.transpose(1, 2)  # (B, T, 2)

    return smoothed




import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class DrivingPlanner(nn.Module):
    def __init__(self, use_depth_aux=False, future_steps=60, history_steps=21, step_dim=3, hidden_dim=128):
        super().__init__()
        self.use_depth_aux = use_depth_aux

        self.cnn_backbone = ImageEncoder()
        self.history_encoder = TrajectoryEncoder()
        self.fusionModule = FusionModule()
        self.decoder = TrajectoryDecoder()

        if use_depth_aux:
            self.depth_encoder = DepthEncoder()
            self.semantic_encoder = SemanticEncoder()

    def forward(self, camera, history, depth_gt):
        B = camera.size(0)

        # (B, 512), (B, )
        image_feat, image_conv = self.cnn_backbone(camera)
        history_feat = self.history_encoder(history)
        fusion_feat = self.fusionModule(image_feat, history_feat)
        pred_future = self.decoder(fusion_feat)

        depth_pred = None
        sem_pred = None
        if self.use_depth_aux:
            depth_pred = self.depth_encoder(image_conv).permute(0, 2, 3, 1)
            sem_pred = self.semantic_encoder(image_conv)            

        return pred_future, depth_pred, sem_pred
    

class DrivingPlanner2(nn.Module):
  def __init__(self, use_depth_aux=False, past_steps=21, step_dim=3, future_steps=60):
      super().__init__()

      self.use_depth_aux = use_depth_aux

      model = models.resnet18(pretrained=models.ResNet18_Weights.DEFAULT)
      self.cnn_backbone = nn.Sequential(*list(model.children())[:-2])
      self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
      self.fc = nn.Linear(512, 512)

      self.ego_encoder = nn.Sequential(
          nn.Linear(step_dim, 16),
          nn.ReLU(),
          nn.Linear(16, 32),
          nn.ReLU()
      )

      history_dim = past_steps * step_dim

      self.decoder = nn.Sequential(
          nn.Linear(512 + 32 + history_dim, 512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Linear(256, future_steps * step_dim)
      )

      if self.use_depth_aux:
        self.depth_encoder = DepthEncoder(input_dim=512)
        self.semantic_encoder = SemanticEncoder(input_dim=512)

  def forward(self, camera, history, depth_gt):
      B = camera.size(0)

      # (B, 512), (B, )
      image_cnn = self.cnn_backbone(camera)
      image_feat = self.fc(self.avg_pool(image_cnn).view(B, -1))
      ego_feat = self.ego_encoder(history[:,-2])
      history_feat = history.view(B, -1)

      concat = torch.cat([image_feat, history_feat, ego_feat], dim=1)
      pred_future = self.decoder(concat).view(B, 60, 3)

      if self.use_depth_aux:
        depth_pred = self.depth_encoder(image_cnn).permute(0, 2, 3, 1)
        sem_pred = self.semantic_encoder(image_cnn)
      else:
        depth_pred = None
        sem_pred = None

      return pred_future, depth_pred, sem_pred
