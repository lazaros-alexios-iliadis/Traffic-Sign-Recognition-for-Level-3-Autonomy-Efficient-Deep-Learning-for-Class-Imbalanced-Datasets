import torch
import torch.nn.functional as F
from model import get_model
from torch.utils.data import DataLoader
from dataset import TrafficSignDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from multiclass_metrics import sba
from scipy.optimize import minimize
from data_preprocessing import load_train_images_from_csv, load_test_images_from_csv, get_transforms, split_data
from config import (train_csv, test_csv, image_folder_train, image_folder_test,
                    batch_size)

# Load the trained models
model_paths = {
    "EfficientNet": "efficientnet.pth",
    "MobileNet": "mobilenet.pth",
    "ProposedCustomCNN": "custom.pth"
}

# Define initial weights
model_weights = {
    "EfficientNet": 0.4,
    "MobileNet": 0.4,
    "ProposedCustomCNN": 0.2
}

num_classes = 43  # Number of traffic sign classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
models = {}
for model_name, model_path in model_paths.items():
    model = get_model(model_name, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    models[model_name] = model

train_images, train_labels = load_train_images_from_csv(train_csv, image_folder_train)
test_images, test_labels = load_test_images_from_csv(test_csv, image_folder_test)

# Split test data into validation and test sets
print("Splitting test data into validation and test sets...")
test_images, val_images, test_labels, val_labels = split_data(test_images, test_labels)
print("Applying data transformations...")
train_transform, test_transform = get_transforms()

# Create datasets
print("Creating datasets...")
train_dataset = TrafficSignDataset(image_paths=train_images, labels=train_labels, transform=train_transform)
val_dataset = TrafficSignDataset(image_paths=val_images, labels=val_labels, transform=test_transform)
test_dataset = TrafficSignDataset(image_paths=test_images, labels=test_labels, transform=test_transform)

# Create data loaders
print("Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Data loaders created with batch size {batch_size}")


def evaluate_ensemble_with_weights(val_loader, model_weights):
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            ensemble_output = torch.zeros((images.size(0), num_classes), device=device)

            for model_name, model in models.items():
                outputs = model(images)
                prob = F.softmax(outputs, dim=1)
                ensemble_output += model_weights[model_name] * prob

            predicted = torch.argmax(ensemble_output, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy


def objective(weights, val_loader):
    w1, w2 = weights
    w3 = 1.0 - (w1 + w2)
    if w3 < 0 or w3 > 1 or w1 == 1.0 or w2 == 1.0 or w3 == 1.0:
        return 1e9  # Penalize invalid weights

    temp_weights = {
        "EfficientNet": w1,
        "MobileNet": w2,
        "ProposedCustomCNN": w3
    }
    acc = evaluate_ensemble_with_weights(val_loader, temp_weights)
    print(f"Testing weights: {temp_weights}, Accuracy: {acc:.2f}%")  # Logging weights during optimization
    return -acc  # We minimize, so negate accuracy


# Constraints: weights sum to 1
constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}

# Bounds: each weight should be between 0 and 1
bounds = [(0, 0.99), (0, 0.99)]


def find_optimal_weights(val_loader):
    print("Starting weight optimization...")
    best_weights = None
    best_acc = -1

    result = minimize(objective, x0=[0.33, 0.33], args=(val_loader,), bounds=bounds, constraints=constraints)

    if result.success:
        w1, w2 = result.x
        w3 = 1.0 - (w1 + w2)
        optimal_weights = {"EfficientNet": w1, "MobileNet": w2, "ProposedCustomCNN": w3}
        best_weights = optimal_weights
        best_acc = -result.fun
        print(f"Optimal Weights Found: {best_weights} with Accuracy: {best_acc:.2f}%")
    else:
        print("Optimization failed, using default weights.")
        best_weights = model_weights

    return best_weights


def evaluate_ensemble(test_loader):
    global model_weights

    # Optimize weights using validation set
    model_weights = find_optimal_weights(test_loader)

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            ensemble_output = torch.zeros((images.size(0), num_classes), device=device)

            for model_name, model in models.items():
                outputs = model(images)
                prob = F.softmax(outputs, dim=1)
                ensemble_output += model_weights[model_name] * prob

            predicted = torch.argmax(ensemble_output, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    sba_score = sba(all_labels, all_predictions)

    print(f'Ensemble Model Accuracy: {accuracy:.2f}%')
    # print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, SBA: {sba_score:.4f}')

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(num_classes),
                yticklabels=np.arange(num_classes))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Ensemble Confusion Matrix")
    plt.show()


evaluate_ensemble(test_loader)
