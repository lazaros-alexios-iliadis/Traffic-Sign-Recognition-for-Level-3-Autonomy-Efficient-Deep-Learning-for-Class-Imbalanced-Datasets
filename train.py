import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from config import num_epochs, learning_rate, numClasses, model_name, warmup_epochs
from model import get_model


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train_model(train_loader, val_loader, device):
    # Load the model from config
    model = get_model(model_name, numClasses)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    
    model.to(device)
    # model.apply(init_weights)  # only for custom models
    # print(model)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def warmup_fn(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # Linear warmup
        else:
            return 0.1 ** ((epoch - warmup_epochs) // 10)  # Step decay after warmup
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)

    def accuracy_fn(y_true, y_pred):
        if len(y_pred.shape) > 1:
            _, y_pred = torch.max(y_pred, 1)
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_true)) * 100
        return acc

    best_val_loss = float('inf')
    best_model_state = None

    # Early stopping parameters
    early_stopping_patience = 5
    early_stopping_counter = 0

    best_val_loss = float('inf')
    best_model_state = None

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += accuracy_fn(labels, outputs)
            loss.backward()
            optimizer.step()

        # Update learning rate
        scheduler.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss = 0
        val_acc = 0
        correct_val_preds = 0  # To accumulate correct predictions
        total_val_samples = 0  # To accumulate total number of samples
        
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy correctly by accumulating correct predictions
                _, preds = torch.max(outputs, 1)
                correct_val_preds += (preds == labels).sum().item()
                total_val_samples += labels.size(0)
        
        # Calculate average loss
        val_loss /= len(val_loader)
        
        # Calculate overall accuracy
        val_acc = (correct_val_preds / total_val_samples) * 100
        
        # Store results
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        # till here

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict()
            # Save the best model
            torch.save(best_model_state, 'best_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs without improvement.")
                break  # Stop training

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies
