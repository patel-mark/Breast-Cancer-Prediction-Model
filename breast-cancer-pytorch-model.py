import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load and Preprocess Data
def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Handle missing values (replace '#' with appropriate strategy)
    df = df.replace('#', np.nan)
    df = df.dropna()
    
    # Convert categorical variables
    categorical_columns = ['Menopause', 'Breast', 'Metastasis', 'History']
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col]).codes
    
    # One-hot encode Breast Quadrant
    df = pd.get_dummies(df, columns=['Breast Quadrant'])
    
    # Prepare features and target
    features = df.drop(['S/N', 'Year', 'Diagnosis Result'], axis=1)
    target = (df['Diagnosis Result'] == 'Malignant').astype(int)
    
    return features, target

# Step 2: Custom PyTorch Dataset
class BreastCancerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features.values)
        self.labels = torch.FloatTensor(labels.values)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Step 3: Neural Network Model
class BreastCancerClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Step 4: Training Pipeline
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_true = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                val_predictions.extend(predictions.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_true, val_predictions)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

# Main Execution
def main():
    # File path
    filepath = 'breast-cancer-dataset.csv'
    
    # Load and preprocess data
    features, target = load_and_preprocess_data(filepath)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_scaled, target, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Create PyTorch Datasets
    train_dataset = BreastCancerDataset(
        pd.DataFrame(X_train), pd.Series(y_train)
    )
    val_dataset = BreastCancerDataset(
        pd.DataFrame(X_val), pd.Series(y_val)
    )
    test_dataset = BreastCancerDataset(
        pd.DataFrame(X_test), pd.Series(y_test)
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Initialize Model
    model = BreastCancerClassifier(input_dim=features.shape[1])
    
    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train Model
    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer
    )
    
    # Evaluation on Test Set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.eval()
    test_predictions = []
    test_true = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = trained_model(features).squeeze()
            predictions = (outputs > 0.5).float()
            test_predictions.extend(predictions.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
    
    # Print Evaluation Metrics
    print("\nTest Set Evaluation:")
    print(classification_report(test_true, test_predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_true, test_predictions))

if __name__ == '__main__':
    main()
