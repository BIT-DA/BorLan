from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class EfficientNetFc(nn.Module):
    def __init__(self, backbone = 'efficientnet-b4', feature_dim=1792, projector_dim=128):
        super(EfficientNetFc, self).__init__()
        print(backbone)
        self.backbone = EfficientNet.from_pretrained(backbone)
        self.proj_fc1 = nn.Linear(feature_dim, projector_dim)
        self.proj_relu = nn.ReLU(inplace=True)
        self.proj_bn = nn.BatchNorm1d(projector_dim)
        self.proj_fc2 = nn.Linear(projector_dim, projector_dim)

    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.backbone._avg_pooling(x)
        if self.backbone._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self.backbone._dropout(x)
            y = self.proj_fc1(x)
            y = self.proj_relu(y)
            y = self.proj_bn(y)
            y = self.proj_fc2(y)
        return y, x
