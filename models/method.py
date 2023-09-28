import torch
import torch.nn as nn

class BorLan(nn.Module):
    def __init__(self, network, backbone, projector_dim=256, feature_dim=256,
                       class_num=200, pretrained=True, pretrained_path=None):
        super(BorLan, self).__init__()
        self.class_num = class_num
        self.backbone = backbone
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path

        # create the encoders
        if 'efficientnet' in self.backbone:
            self.encoder = network(backbone=self.backbone, feature_dim=feature_dim, projector_dim=projector_dim)
        else:
            self.encoder = network(projector_dim=projector_dim)

        self.load_pretrained(network)

    def forward(self, img):
        """
        emb: projector output, for language semantic space alignment
        feat: encoder output, for classification
        """
        emb, feat = self.encoder(img)  # emb (N x projector_dim)
        emb = nn.functional.normalize(emb, dim=1)

        return emb, feat

    def load_pretrained(self, network):
        if 'resnet' in self.backbone:
            q = network(projector_dim=1000, pretrained=self.pretrained)
            q.proj_fc1 = self.encoder.proj_fc1
            q.proj_bn = self.encoder.proj_bn
            q.proj_relu = self.encoder.proj_relu
            q.proj_fc2 = self.encoder.proj_fc2
            self.encoder = q
        elif 'densenet' in self.backbone:
            q = network(projector_dim=1000, pretrained=self.pretrained)
            q.classifier = self.encoder.classifier
            self.encoder = q

    def inference(self, img):
        _, feat = self.encoder(img)
        return feat
