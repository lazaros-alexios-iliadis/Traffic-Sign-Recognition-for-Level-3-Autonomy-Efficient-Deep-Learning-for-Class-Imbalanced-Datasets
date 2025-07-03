import torch
from torch.utils.data import DataLoader
from data_preprocessing import load_train_images_from_csv, load_test_images_from_csv, get_transforms, split_data
from dataset import TrafficSignDataset
from train import train_model
from evaluate import evaluate_model
from config import (train_csv, test_csv, image_folder_train, image_folder_test,
                    batch_size)
import matplotlib.pyplot as plt
import time


def main():
    # Load and preprocess data
    print("Loading and preprocessing training and testing data...")
    train_images, train_labels = load_train_images_from_csv(train_csv, image_folder_train)
    test_images, test_labels = load_test_images_from_csv(test_csv, image_folder_test)

    # Split test data into validation and test sets
    print("Splitting test data into validation and test sets...")
    test_images, val_images, test_labels, val_labels = split_data(test_images, test_labels)

    # Get data transformations
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

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Start the timer
    print("Starting model training...")
    start_time = time.time()

    # Train the model
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(train_loader, val_loader, device)
    print("Model training completed.")

    # Evaluate the model
    print("Evaluating the model on the test set...")
    criterion = torch.nn.CrossEntropyLoss()
    evaluate_model(model, test_loader, device, criterion)
    print("Model evaluation completed.")

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time in seconds
    print(f"Total training and evaluation time: {elapsed_time:.2f} seconds")

    save_path1 = 'accuracy.png'
    save_path2 = 'loss.png'

    # Plot training and validation losses
    print("Plotting and saving loss graphs...")

    # Plot training and validation accuracies
    # Accuracy Plot
    plt.figure(figsize=(6, 4), dpi=300)  # Set figure size and high DPI for publication
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Training Accuracy", linewidth=2, marker="o",
             markersize=4)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy", linewidth=2, marker="s",
             markersize=4)

    # Labels & Title
    plt.xlabel("Epochs", fontsize=10, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=10, fontweight="bold")  # Change to "Accuracy" if values are between 0 and 1
    plt.title("ResNet18: Accuracy", fontsize=10, fontweight="bold")

    # Grid & Ticks
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Legend
    plt.legend(fontsize=10, loc="lower right", frameon=True)

    # Save and Show
    plt.tight_layout()
    plt.savefig(save_path1, dpi=300, bbox_inches="tight")
    plt.show()

    # Loss Plot
    plt.figure(figsize=(6, 4), dpi=300)  # Set figure size and high DPI for publication
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", linewidth=2, marker="o",
             markersize=4)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", linewidth=2, marker="s", markersize=4)

    # Labels & Title
    plt.xlabel("Epochs", fontsize=10, fontweight="bold")
    plt.ylabel("Loss", fontsize=10, fontweight="bold")
    plt.title("ResNet18: Loss", fontsize=10, fontweight="bold")

    # Grid & Ticks
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Legend
    plt.legend(fontsize=10, loc="upper right", frameon=True)  # Loss is usually best viewed with legend in upper right

    # Save and Show
    plt.tight_layout()
    plt.savefig(save_path2, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
