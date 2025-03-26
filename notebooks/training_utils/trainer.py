import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

from .losses import FocalLoss
from .models import TreeClassifier
from .torch_datasets import TreeDataset
from .visualization import plot_training_history, plot_confusion_matrix


# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def create_weighted_sampler(y):
    """Create a weighted sampler for handling class imbalance."""
    class_counts = np.bincount(y)
    weights = 1.0 / class_counts

    sample_weights = [weights[cls] for cls in y]
    sample_weights = torch.DoubleTensor(sample_weights)  # or FloatTensor

    return WeightedRandomSampler(weights=sample_weights,
                                 num_samples=len(sample_weights),
                                 replacement=True)


def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(train_loader), correct / total


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on test data."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(test_loader), correct / total, all_predictions, all_labels


def train_tree_classifier(X_train, y_train, X_test, y_test):

    X_train = X_train.values
    X_test = X_test.values

    train_dataset = TreeDataset(X_train, y_train)
    test_dataset = TreeDataset(X_test, y_test)

    sampler = create_weighted_sampler(y_train)
    # custom DataLoader with cass weight samples as we have imbalance in classes
    # 4096 batch size as we are dealing with tabular data
    train_loader = DataLoader(train_dataset, batch_size=4096, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")

    # Create and compile the model
    print("Creating model...")
    num_classes = len(np.unique(y_train))
    # decisions for model architecture can be found in models.py
    model = TreeClassifier(X_test.shape[1], num_classes).to(device)
    print(model)

    # FocalLoss as we have imbalance in classes
    criterion = FocalLoss(alpha=1, gamma=2)  
    optimizer = optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.01)
    # ReduceLROnPlateau as for better convergence
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training loop
    print("Training model...")
    num_epochs = 30
    best_val_loss = float('inf')
    # Early stopping patience for 10 as we have ReducedLROnPlateau scheduler
    patience = 10
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)

        # Evaluate
        val_loss, val_acc, _, _ = evaluate_model(model, test_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), '../models/tree_health_classifier.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Plot training history
    print("Plotting training history...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    # Load best model and make predictions
    print("Making predictions...")
    model.load_state_dict(torch.load('../models/tree_health_classifier.pth'))
    _, _, y_pred, y_true = evaluate_model(model, test_loader, criterion, device)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    train_tree_classifier()
