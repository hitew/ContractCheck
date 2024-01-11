import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score,matthews_corrcoef
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read balanced data
data_path = './data/train_data/elmo_balanced_data.csv'
balanced_data = pd.read_csv(data_path)

# Split features and labels
X = balanced_data.drop('label', axis=1).values
y = balanced_data['label'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define CNN and BiGRU model
class CNNBiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(CNNBiGRU, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bigru = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  
        x, _ = self.bigru(x)
        x = x[:, -1, :]  
        x = self.fc(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
input_dim = X_train.shape[1]  
hidden_dim = 150
num_layers = 2
num_classes = len(np.unique(y))
model = CNNBiGRU(input_dim, hidden_dim, num_layers, num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
best_accuracy = 0.0
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Calculate training accuracy
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)

    # Calculate testing accuracy
    correct = 0
    total = 0
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')


    if test_accuracy > best_accuracy:
       best_accuracy = test_accuracy
       model_file_name = f'./Results/Models/cnn_bigru_models/elmo_model_epoch_{epoch}_acc_{best_accuracy:.4f}.pt'
       torch.save(model.state_dict(), model_file_name)
       best_model_file_name = model_file_name  

# Plot accuracy during training
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Process Accuracy')
plt.legend()
plt.savefig('./Results/Graphs/elmo_cnn_bigru_results.png')
plt.show()


# Use the best model for prediction
model.load_state_dict(torch.load(best_model_file_name))

model.eval()
with torch.no_grad():
    y_pred = []
    for data in DataLoader(test_dataset, batch_size=batch_size):
        data = data[0].to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())

# Calculate metrics on the test set
report = classification_report(y_test, y_pred)
print(report)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

precision = precision_score(y_test, y_pred, average=None)
print("Precision for Each Class:")
print(precision)

recall = recall_score(y_test, y_pred, average=None)
print("Recall for Each Class:")
print(recall)

# Calculate false positive rate
fpr = cm.sum(axis=0) - np.diag(cm)
fpr = fpr / cm.sum(axis=0)
print("False Positive Rate for Each Class:")
print(fpr)

fpr_total =  cm[1][0] / (cm[1][0] + cm[1][1])
print(fpr_total)

mcc = matthews_corrcoef(y_test, y_pred)


print("Total FPR:", fpr_total)
print("Total MCC:", mcc)