import torchvision.models as models
import torch.nn as nn
from CustomCNN import SimpleCNN, MicroVGG, ProposedCustomCNN


def get_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg11':
        model = models.vgg11(weights='IMAGENET1K_V1')
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
    elif model_name == 'convnext':
        model = models.convnext_small(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_ftrs, out_features=num_classes)
        )
    elif model_name == 'EfficientNet':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'MobileNet':
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'simple_cnn':
        model = SimpleCNN(num_classes=num_classes)
    elif model_name == 'MicroVGG':
        model = MicroVGG(num_classes=num_classes)
    elif model_name == 'custom_cnn':
        model = ProposedCustomCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
