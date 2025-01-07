import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# Custom CNN with more layers and regularization for FashionMNIST
class CustomCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Accuracy Checker
def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()  # Evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()  # Back to training mode
    return 100 * float(num_correct) / float(num_samples)


# Evaluate Model
def evaluate_model(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            _, preds = scores.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_acc_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward and Backward Pass
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Adjust Learning Rate
        scheduler.step()

        # Check Accuracy
        train_acc = check_accuracy(train_loader, model, device)
        val_acc = check_accuracy(val_loader, model, device)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

    # Visualize accuracy
    plt.plot(range(num_epochs), train_acc_list, label="Train Accuracy")
    plt.plot(range(num_epochs), val_acc_list, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.show()


# Visualize results for a few test images
def visualize_results(model, test_loader, device):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    fig = plt.figure(figsize=(12, 6))
    for i in range(6):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(images[i].cpu().squeeze(), cmap="gray")
        ax.set_title(f"Pred: {preds[i].item()} / True: {labels[i].item()}")
        ax.axis("off")
    plt.show()


# Main Function
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    in_channels = 1 # FashinoMNIST has 1 channel (grayscale)
    num_classes = 10  # FashionMNIST has 10 classes (clothing categories)
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 5

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    # Load FashionMNIST dataset
    train_dataset = datasets.FashionMNIST(root='dataset/', train=True, transform=train_transform, download=True)
    test_dataset = datasets.FashionMNIST(root='dataset/', train=False, transform=test_transform, download=True)

    # Splitting into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss, optimizer, and scheduler
    model = CustomCNN(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

    # Final Accuracy Check
    test_acc = check_accuracy(test_loader, model, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # Evaluate Model
    evaluate_model(test_loader, model, device)

    # Visualize results
    visualize_results(model, test_loader, device)


# Argument Parser for Dynamic Hyperparameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--hidden_size", type=int, default=50, help="Number of neurons in hidden layer")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for regularization")
    args = parser.parse_args()

    main(args)
