import torch
import torch.nn as nn
import os
from torchvision import datasets
from torchvision import transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]) #making every image the same size and converting it to a tensor
test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]) #making every image the same size and converting it to a tensor

train_dataset = datasets.ImageFolder(root = 'dataset/train', transform = train_transform)
val_dataset = datasets.ImageFolder(root = 'dataset/val', transform = test_transform)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)

model = models.resnet18(weights = "DEFAULT") #Using a pre-trained ResNet-18 model with default weights
model.fc = nn.Linear(model.fc.in_features, 2) #Modifying the final fully

for parameter in model.parameters():
    parameter.requires_grad = False #Allowing all parameters to be trainable
    
for parameter in model.layer4.parameters():
    parameter.requires_grad = True #Allowing parameters in layer 4 to be trainable
    
for parameter in model.fc.parameters():
    parameter.requires_grad = True #Ensuring the final layer is trainable
    
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([{"params": model.layer4.parameters(), "lr": 0.0001}, {"params": model.fc.parameters(), "lr":0.001 }]) #Using Adam optimizer for the final layer with a learning rate of 0.001

for epoch in range(5):
    model.train() #Setting the model to training mode
    for images, labels in train_loader:
        predictions = model(images) 
        loss = loss_function(predictions, labels)
        optimizer.zero_grad() #Clearing previous gradients
        loss.backward() #Computing gradients
        optimizer.step() #Updating weights or model parameters
    
    print("Epoch:", epoch, "Loss:", loss.item())
    
model.eval() #Setting the model to evaluation mode
    
correct = 0
total = 0
    
with torch.no_grad():
    for images, labels in val_loader:
        predictions = model(images)
        predicted_labels = predictions.argmax(dim = 1)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

print("Validation Accuracy:", correct/total)

test_dataset = datasets.ImageFolder("dataset/test", transform = test_transform)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

model.eval()

all_predictions = []
all_labels = []
all_paths = []

with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        predicted_labels = predictions.argmax(dim = 1)
        
        all_predictions.extend(predicted_labels.tolist())
        all_labels.extend(labels.tolist())
        
        for path in test_loader.dataset.samples[
            len(all_paths):len(all_paths) + len(images)
        ]: all_paths.append(path[0])

test_accuracy = sum(
    prediction == label
    for prediction, label in zip(all_predictions, all_labels)
) / len(all_labels)

print("Test Accuracy:", test_accuracy)

matrix = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(matrix)

print("\nFalse Positives (no_smoke predicted as smoke):")

for path, prediction, label in zip(
    all_paths,
    all_predictions,
    all_labels
):
    if label==0 and prediction==1:
        print(path)

torch.save(model.state_dict(), "wildfire_model.pth")