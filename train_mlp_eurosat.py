
"""
start with 2 hidden layers and 2048 units
Use ReLU activations
Adam optimizer
Learning rate = 0.001
start with 20 epochs
batch size?
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse # To pass in block_idx and timestep if needed

# Import your dataloader and category list
from src.flux.eurosat_dataloader import get_eurosat_dataloader, get_eurosat_categories

# Import the feature extractor
from src.flux.feat_flux import Featurizer4Eval 

# ---------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------
DATA_ROOT = "/lustre/isaac24/scratch/jdosch1/DeepLearning/datasets/EuroSAT"
BATCH_SIZE = 1 # MUST be 1 for this specific implementation of Featurizer4Eval
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 10 
HIDDEN_DIM = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# Define the MLP
# ---------------------------------------------------------
class EuroSAT_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------
# Training Setup
# ---------------------------------------------------------
def train_mlp():
    # Setup a dummy args object to pass to the forward pass
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()

    print("Loading datasets...")
    # NOTE: The batch size must be 1 because Featurizer4Eval hardcodes `.unsqueeze(0)` internally.
    train_loader = get_eurosat_dataloader(root=DATA_ROOT, split="train", batch_size=BATCH_SIZE, shuffle=True)
    
    # Get the categories for prompt preparation
    cat_list = get_eurosat_categories()

    print("Loading frozen FLUX backbone (this may take a minute)...")
    featurizer = Featurizer4Eval(flux_id="flux-dev", cat_list=cat_list)

    # Determine Feature Dimensions via a dummy pass
    sample_batch = next(iter(train_loader))
    dummy_img = sample_batch["img"][0].to(device) # Featurizer expects [C, H, W]
    dummy_cat = sample_batch["class_name"][0]
    
    with torch.no_grad(): 
        # Unpack the tuple: we only want the features, not the mod parameters
        dummy_features, _ = featurizer.forward(args=args, img_tensor=dummy_img, category=dummy_cat, block_idx=[1])
        
        # Output is [1, C, H, W]. Global average pool over H and W.
        dummy_features = dummy_features.mean(dim=[2, 3]) 
    
    feature_dim = dummy_features.shape[1]
    print(f"Detected FLUX representation dimension: {feature_dim}")

    mlp = EuroSAT_MLP(input_dim=feature_dim, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=LEARNING_RATE)

    # ---------------------------------------------------------
    # The Training Loop
    # ---------------------------------------------------------
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        mlp.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Loop over the training data
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            # Dataloader gives us a batch dimension, but Featurizer4Eval wants a single image [C, H, W]
            image = batch["img"][0].to(device) 
            label = batch["label"].to(device) # This is a tensor like [0]
            category = batch["class_name"][0] # This is a string

            with torch.no_grad():
                # Pass image and text category to get features
                features, _ = featurizer.forward(args=args, img_tensor=image, category=category, block_idx=[1])
                features = features.mean(dim=[2, 3]) # [1, C]

            features = features.to(torch.float32)
            optimizer.zero_grad()
            outputs = mlp(features) # Output shape will be [1, 10]

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            correct += (predicted == label).sum().item()

        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Train Acc: {epoch_acc:.2f}%")

if __name__ == "__main__":
    train_mlp()