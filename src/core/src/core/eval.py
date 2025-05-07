import torch
from sklearn.metrics import accuracy_score


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model: Trained MFA Conformer model.
        test_loader: Voxceleb DataLoader for the test dataset.
        device: Device to run the evaluation (CPU or GPU).

    Returns:
        float: Accuracy score on the test set.
    """
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get class predictions

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute the accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


# Example Usage
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("model_checkpoint_best.pth")  # Load the best model checkpoint
    model = model["model_state_dict"]  # Adjust if checkpoint includes state_dict
    model.to(device)

    # Evaluate the model
    evaluate_model(model, voxceleb_dataloader_test, device)
