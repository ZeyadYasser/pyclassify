import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class SqueezeNet(nn.Module):
    def __init__(self, classes):
        super(SqueezeNet, self).__init__()

        self.classes = classes
        self.num_classes = len(classes)
        self.model = models.squeezenet1_0(pretrained=True)
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Conv2d(512, self.num_classes, kernel_size=1),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.model.features(x)
        x = self.classifier(x)
        return x.view(-1, self.num_classes)