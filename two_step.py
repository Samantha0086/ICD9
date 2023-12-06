from gensim.models import Word2Vec

# Example ICD-9 sequences (replace with your data)
icd9_sequences = [['123', '456', '789'], ['234', '567', '890']]  

# Train a skip-gram model
model = Word2Vec(icd9_sequences, size=100, window=5, min_count=1, workers=4, sg=1)


import numpy as np

def get_embeddings(icd9_codes, embeddings, embedding_size):
    return np.array([embeddings[code] if code in embeddings else np.zeros(embedding_size) for code in icd9_codes])

# Convert ICD-9 codes to embeddings
# This will depend on how your data is structured. This is just a placeholder.
icd9_embeddings = np.array([get_embeddings(seq, model.wv, 100) for seq in icd9_sequences])

# Other features preparation (placeholder)
other_features = np.array([[1.5, 2.3], [0.4, 0.7]])  # Replace with your data

# Assuming binary labels
labels = np.array([0, 1])  # Replace with your data

class MortalityPredictionModel(nn.Module):
    def __init__(self, embedding_size, num_other_features, num_classes):
        super(MortalityPredictionModel, self).__init__()
        # Define layers of the CNN
        self.conv1 = nn.Conv1d(in_channels=embedding_size, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(64 + num_other_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_icd9, x_other):
        # Forward pass through CNN layers
        x_icd9 = F.relu(self.conv1(x_icd9))
        # Flatten the output for the fully connected layer
        x_icd9 = torch.flatten(x_icd9, 1)
        # Combine with other features
        x_combined = torch.cat((x_icd9, x_other), dim=1)
        # Forward pass through fully connected layers
        x = F.relu(self.fc1(x_combined))
        x = self.fc2(x)
        return x


# Assuming the use of a simple custom dataset
class MortalityDataset(torch.utils.data.Dataset):
    def __init__(self, icd9_embeddings, other_features, labels):
        self.icd9_embeddings = icd9_embeddings
        self.other_features = other_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.icd9_embeddings[idx], self.other_features[idx], self.labels[idx]

# Create dataset and dataloader
dataset = MortalityDataset(icd9_embeddings, other_features, labels)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

model = MortalityPredictionModel(embedding_size=100, num_other_features=2, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(1):  # Set the number of epochs
    for icd9_batch, other_features_batch, labels_batch in train_loader:
        icd9_batch = icd9_batch.float().transpose(1, 2)  # Adjust the shape for Conv1d
        other_features_batch = torch.tensor(other_features_batch).float()
        labels_batch = torch.tensor(labels_batch).long()

        # Forward pass
        outputs = model(icd9_batch, other_features_batch)
        loss = F.cross_entropy(outputs, labels_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Assuming you have a validation dataset prepared
val_icd9_embeddings = ...  # Validation ICD-9 embeddings
val_other_features = ...   # Validation other features
val_labels = ...           # Validation labels

val_dataset = MortalityDataset(val_icd9_embeddings, val_other_features, val_labels)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

# Switch model to evaluation mode
model.eval()

# Track variables for accuracy calculation
total = 0
correct = 0

with torch.no_grad():  # No need to track gradients during validation
    for icd9_batch, other_features_batch, labels_batch in val_loader:
        icd9_batch = icd9_batch.float().transpose(1, 2)  # Adjust shape for Conv1d
        other_features_batch = torch.tensor(other_features_batch).float()
        labels_batch = torch.tensor(labels_batch).long()

        # Forward pass
        outputs = model(icd9_batch, other_features_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')