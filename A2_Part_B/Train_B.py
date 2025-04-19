import argparse
import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

def main(args):
    # Initialize WandB
    wandb.init(project=args.wandb_project, config=args)
    
    # Data transforms with ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(f"{args.data_dir}/train", transform)
    val_dataset = datasets.ImageFolder(f"{args.data_dir}/val", transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classifier
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(512, 10)
    )
    
    # Move model to device after modification
    model = model.to(device)
    
    # Training setup (only train classifier)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            wandb.log({"train_loss": loss.item()})
        
        # Validation
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / len(val_loader.dataset)
        wandb.log({
            "epoch": epoch+1,
            "val_loss": val_loss/len(val_loader),
            "val_acc": val_acc
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 model")
    parser.add_argument("--data_dir", default="inaturalist_12K", help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--wandb_project", default="DL-Assignment2-ResNet", 
                       help="W&B project name")
    args = parser.parse_args()
    
    main(args)