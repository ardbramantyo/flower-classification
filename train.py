import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_datasets = {
        'train': ImageFolder(root=train_dir, transform=data_transforms),
        'val': ImageFolder(root=valid_dir, transform=data_transforms),
        'test': ImageFolder(root=test_dir, transform=data_transforms)
    }
    
    return image_datasets

def create_dataloaders(image_datasets, batch_size, num_workers):
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True, drop_last=True)
    }
    return dataloaders

def initialize_model(arch, num_classes):
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif arch == "resnet":
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Invalid architecture: {arch}")
    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    model.to(device)
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        
        for images, labels in dataloaders['train']:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy += (predicted == labels).sum().item()

        train_loss = train_loss / len(dataloaders['train'].dataset)
        train_accuracy = train_accuracy / len(dataloaders['train'].dataset)

        val_loss, val_accuracy = evaluate_model(model, dataloaders['val'], criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_to_idx': dataloaders['train'].dataset.class_to_idx
            }
            torch.save(checkpoint, 'best_model_checkpoint.pth')
    
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    loss = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            accuracy += (predicted == labels).sum().item()

    loss = loss / len(dataloader.dataset)
    accuracy = accuracy / len(dataloader.dataset)

    return loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Train a deep learning model.')
    parser.add_argument('data_directory', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='resnet', choices=['vgg13', 'resnet'],
                        help='Choose the architecture (vgg13 or resnet)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()
    
    # Load data
    data_dir = args.data_directory
    image_datasets = load_data(data_dir)
    
    # Create dataloaders
    dataloaders = create_dataloaders(image_datasets, args.batch_size, args.num_workers)
    
    # Check GPU availability
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = initialize_model(args.arch, len(image_datasets['train'].classes))
    
    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    
    # Train the model
    train_model(model, dataloaders, criterion, optimizer, args.epochs, device)
    
if __name__ == "__main__":
    main()