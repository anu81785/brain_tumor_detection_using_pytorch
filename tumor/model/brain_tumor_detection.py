import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
class CustomResNet50:
    def __init__(self, num_classes, pretrained=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = self._create_model()

    def _create_model(self):
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Set all parameters as trainable
        for param in model.parameters():
            param.requires_grad = True

        # Get input size of the fully connected layer
        inputs = model.fc.in_features

        # Redefine the fully connected (fc) layer for classification
        model.fc = nn.Sequential(
            nn.Linear(inputs, 2048),
            nn.SELU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.SELU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, self.num_classes),
            nn.LogSigmoid()
        )

        # Set all parameters of the model as trainable
        for name, child in model.named_children():
            for name2, params in child.named_parameters():
                params.requires_grad = True

        model.to(self.device)
        return model

    def get_model(self):
        return self.model

    def print_architecture(self):
        print(self.model)
 