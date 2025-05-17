import torch
import torch.nn as nn
import torchvision.models as models

class DepthDecoder(nn.Module):
    def __init__(self, in_channels=2048):
        super().__init__()

        self.up1 = self._upsample_block(in_channels, 512)     # 6x9 → 12x18
        self.up2 = self._upsample_block(512, 256)             # 12x18 → 25x37
        self.up3 = self._upsample_block(256, 128)             # 25x37 → 50x75
        self.up4 = self._upsample_block(128, 64)              # 50x75 → 100x150
        self.up5 = self._upsample_block(64, 32)               # 100x150 → 200x300

        self.up6 = nn.Upsample(size=(200,300), mode='bilinear', align_corners=False)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)  # Output: (B, 1, 200, 300)

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up1(x)       # 12x18
        x = self.up2(x)       # ~25x37
        x = self.up3(x)       # ~50x75
        x = self.up4(x)       # ~100x150
        x = self.up5(x)       # ~200x300
        x = self.up6(x)
        x = self.out_conv(x)  # (B, 1, 200, 300)
        return x

class SemanticDecoder(nn.Module):
    def __init__(self, in_channels=2048, num_classes=15):
        super().__init__()

        self.up1 = self._upsample_block(in_channels, 512)     # 6x9 → 12x18
        self.up2 = self._upsample_block(512, 256)             # 12x18 → ~25x37
        self.up3 = self._upsample_block(256, 128)             # ~25x37 → ~50x75
        self.up4 = self._upsample_block(128, 64)              # ~50x75 → ~100x150
        self.up5 = self._upsample_block(64, 32)               # ~100x150 → ~200x300

        self.up6 = nn.Upsample(size=(200,300), mode='bilinear', align_corners=False)
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)  # (B, 3, 200, 300)

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)
        x = self.out_conv(x)  # Final output: (B, 3, 200, 300)
        return x

class DrivingPlanner(nn.Module):
    def __init__(self, use_depth_aux=False, use_semantic_aux=False, past_steps=21, step_dim=3, future_steps=60):
        super().__init__()

        self.use_depth_aux = use_depth_aux
        self.use_semantic_aux = use_semantic_aux

        model = models.resnet50(pretrained=models.ResNet50_Weights.DEFAULT)
        self.cnn_backbone = nn.Sequential(*list(model.children())[:-2])
        cnn_backbone_out_dim = model.fc.in_features
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cnn_backbone_out_dim, 512)

        history_dim = past_steps * step_dim

        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, future_steps * step_dim)
        )

        if self.use_depth_aux:
            self.depth_encoder = DepthDecoder(in_channels=cnn_backbone_out_dim)
        else:
            self.depth_encoder = None

        if self.use_semantic_aux:
            self.semantic_encoder = SemanticDecoder(in_channels=cnn_backbone_out_dim)
        else:
            self.semantic_encoder = None

    def forward(self, camera, history):
        B = camera.size(0)

        # (B, 512), (B, )
        image_cnn = self.cnn_backbone(camera)
        image_feat = self.fc(self.avg_pool(image_cnn).view(B, -1))
        history_feat = self.history_encoder(history.view(B, -1))

        concat = torch.cat([image_feat, history_feat], dim=1)
        pred_future = self.decoder(concat).view(B, 60, 3)

        if self.use_depth_aux:
            depth_pred = self.depth_encoder(image_cnn).permute(0, 2, 3, 1)
        else:
            depth_pred = None

        if self.use_semantic_aux:
            sem_pred = self.semantic_encoder(image_cnn)
        else:
            sem_pred = None

        return pred_future, depth_pred, sem_pred
