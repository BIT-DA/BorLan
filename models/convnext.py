from transformers import ConvNextModel
import torch.nn as nn

class ConvNextFc(nn.Module):
    def __init__(self, backbone = 'convnext-tiny-224', feature_dim=768, projector_dim=128):
        super(ConvNextFc, self).__init__()
        print(backbone)
        self.backbone = ConvNextModel.from_pretrained("facebook/"+backbone)
        self.fc = nn.Linear(feature_dim, projector_dim)

    def forward(self, x):
        x = self.backbone(x).pooler_output
        y = self.fc(x)
        return y, x