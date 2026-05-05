import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BATCH_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 10 
HIDDEN_DIM = 2048

# EuroSAT classes are typically sorted alphabetically by default in PyTorch
EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# Define the MLP
# ---------------------------------------------------------
class EuroSAT_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Second hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Output layer
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------
# Training & Evaluation Loop
# ---------------------------------------------------------
def train_eval_mlp():
    print("Loading cached features from disk...")
    
    # Load BOTH train and test splits
    train_data = torch.load('eurosat_flux_features_train.pt')
    test_data = torch.load('eurosat_flux_features_test.pt')
    
    train_features, train_labels = train_data['features'], train_data['labels']
    test_features, test_labels = test_data['features'], test_data['labels']
    
    feature_dim = train_features.shape[1]
    print(f"Loaded Train Features: {train_features.shape[0]} images, {feature_dim} dimensions")
    print(f"Loaded Test Features: {test_features.shape[0]} images, {feature_dim} dimensions")
    
    # Create DataLoaders
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize MLP
    mlp = EuroSAT_MLP(input_dim=feature_dim, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting training loop...")
    for epoch in range(NUM_EPOCHS):
        
        # --- TRAINING PHASE ---
        mlp.train()
        total_loss = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = mlp(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # --- EVALUATION PHASE ---
        mlp.eval()
        correct = 0
        total = 0
        
        # Lists to store labels and predictions for F1 and Confusion Matrix
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = mlp(batch_features)
                _, predicted = torch.max(outputs.data, 1) # Get the index of the highest logit
                
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                # Store predictions and true labels (move to CPU for scikit-learn)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
        test_accuracy = 100 * correct / total
        
        # Calculate Macro F1 Score
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Acc: {test_accuracy:.2f}% | Test F1: {macro_f1:.4f}")

        # --- PLOT CONFUSION MATRIX ON FINAL EPOCH ---
        if epoch == NUM_EPOCHS - 1:
            print("\nGenerating confusion matrix for final epoch...")
            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EUROSAT_CLASSES)
            
            # Recreate the styling from the attached image
            fig, ax = plt.subplots(figsize=(12, 10))
            disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='d')
            
            plt.title("EuroSAT Confusion Matrix (Final Epoch)", fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            save_path = "eurosat_confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to '{save_path}'")

if __name__ == "__main__":
    train_eval_mlp()