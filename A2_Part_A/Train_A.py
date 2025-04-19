import argparse
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

class CustomCNN(nn.Module):
    """Custom CNN with configurable architecture"""
    def __init__(self, num_filters=[32,64,128], filter_sizes=[3,3,3], 
                 activation='relu', dense_units=128, dropout=0.2, use_bn=True):
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            *self._conv_block(3, num_filters[0], filter_sizes[0], activation, use_bn),
            *self._conv_block(num_filters[0], num_filters[1], filter_sizes[1], activation, use_bn),
            *self._conv_block(num_filters[1], num_filters[2], filter_sizes[2], activation, use_bn)
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._calc_features(), dense_units),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 10)
        )
    
    def _conv_block(self, in_c, out_c, kernel_size, activation, use_bn):
        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        block = [
            nn.Conv2d(in_c, out_c, kernel_size, padding=1),
            nn.MaxPool2d(2),
            activations[activation]
        ]
        if use_bn:
            block.insert(2, nn.BatchNorm2d(out_c))
        return block

    def _calc_features(self):
        dummy = torch.randn(1, 3, 224, 224)
        return self.features(dummy).view(1, -1).shape[1]

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train(args):
    # Initialize WandB
    wandb.init(project=args.wandb_project, config=args)
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    full_dataset = datasets.ImageFolder(f"{args.data_dir}/train", transform)
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        stratify=full_dataset.targets
    )
    
    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN(
        num_filters=args.num_filters,
        filter_sizes=args.filter_sizes,
        activation=args.activation,
        dense_units=args.dense_units,
        dropout=args.dropout,
        use_bn=args.use_bn
    ).to(device)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
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
    parser = argparse.ArgumentParser(description="Train custom CNN model")
    parser.add_argument("--data_dir", default="inaturalist_12K", help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_filters", nargs=3, type=int, default=[32,64,128], 
                       help="Number of filters for each conv layer")
    parser.add_argument("--filter_sizes", nargs=3, type=int, default=[3,3,3], 
                       help="Kernel sizes for each conv layer")
    parser.add_argument("--activation", choices=['relu','elu','selu'], default='relu',
                       help="Activation function")
    parser.add_argument("--dense_units", type=int, default=256, help="Dense layer units")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--use_bn", action='store_true', help="Use batch normalization")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--wandb_project", default="DL-Assignment2-CNN", 
                       help="W&B project name")
    args = parser.parse_args()
    
    train(args)