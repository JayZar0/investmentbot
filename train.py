import torch

## This file will be used to contain all the training and validation functions
## for the llm

# This function is the training function that will use the dataset provided
def train_llm(dataloader, model, criterion, optimizer, epoch, num_epoch):
    epoch_loss = 0
    model.train()
    for inputs, labels in dataloader:  # Assume `dataloader` is defined
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
    epoch_loss /= len(dataloader.dataset)
    print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {epoch_loss:.4f}')

# This function is the validation function that wil be used the make sure
# that the training is accurate to what teh llm need to done
def validate_training(test_dataloader, model):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_dataloader:  # Assume `test_dataloader` is defined
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total:.2f}%')
