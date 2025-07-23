import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from model import SHRNet
from dataset import BrainDataset
from loss import OffsetLoss


num_epochs = 500
batch_size = 16
lr = 0.001
dataset = BrainDataset()
model = SHRNet()
criterion = OffsetLoss()

# Divide the training set and the test set
train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
train_dataset = torch.utils.data.Subset(dataset, train_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

# Create a data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, the loss function and the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#  list used for storing indicators
train_losses = []
f1_scores = []
auc_scores = []
accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')

    # test model
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = correct / total
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    accuracies.append(acc)
    f1_scores.append(f1)
    auc_scores.append(auc)

    print(f'Epoch {epoch+1}/{num_epochs}, Acc: {acc * 100:.2f}%, F1: {f1:.4f}, AUC: {auc:.4f}')