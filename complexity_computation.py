import torchvision.models as models
from ptflops import get_model_complexity_info
import pandas as pd
from CustomCNN import MicroVGG

# Define the models
models_dict = {
    "ResNet18": models.resnet18(),
    "EfficientNet-B0": models.efficientnet_b0(),
    "MobileNetV3-Small": models.mobilenet_v3_small(),
    "MicroVGG": MicroVGG(num_classes=43)
}

# Compute model complexity
complexity_results = {"Model": [], "Parameters (M)": [], "FLOPs (G)": []}

for model_name, model in models_dict.items():
    macs, params = get_model_complexity_info(model, (3, 128, 128), as_strings=False, print_per_layer_stat=False)
    complexity_results["Model"].append(model_name)
    complexity_results["Parameters (M)"].append(params / 1e6)  # Convert to Millions
    complexity_results["FLOPs (G)"].append(macs / 1e9)  # Convert to GigaFLOPs

# Display results
df_complexity = pd.DataFrame(complexity_results)
print(df_complexity)
