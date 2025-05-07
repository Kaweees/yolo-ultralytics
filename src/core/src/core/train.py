# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss, correct, total = 0, 0, 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        epoch_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = epoch_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Metrics
            epoch_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = epoch_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, file_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        file_path,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    checkpoint_path,
):
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Training phase
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # Validation phase
        val_loss, val_accuracy = validate_one_epoch(
            model, val_loader, criterion, device
        )
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # U can add here a criteria to check the accuracy and the loss of the validation set..

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, f"{checkpoint_path}_epoch_{epoch}.pth")


# Example Usage
if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(
        model,
        voxceleb_dataloader_train,
        voxceleb_dataloader_val,
        criterion,
        optimizer,
        num_epochs=5,
        device=device,
        checkpoint_path="model_checkpoint",
    )
