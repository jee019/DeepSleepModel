import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import transformers
from transformers import CvtForImageClassification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the pre-trained model
model_name = 'microsoft/cvt-13'
model = CvtForImageClassification.from_pretrained(model_name)

# Modify the number of output classes
num_classes = 4
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the dataset and define the optimizer and loss function
transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = torchvision.datasets.ImageFolder('./train', transform=transform)
val_dataset = torchvision.datasets.ImageFolder('./val', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Train the model
num_epochs = 9
train_losses = []
train_acc = []
val_acc = []

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = 100 * correct / total
    train_acc.append(train_accuracy)

    # Evaluate the model on validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    val_acc.append(val_accuracy)

    print('[Epoch %d] Train Loss: %.4f, Train Accuracy: %.2f%%, Val Accuracy: %.2f%%' % (epoch + 1, train_loss, train_accuracy, val_accuracy))

test_dataset = torchvision.datasets.ImageFolder('./test', transform=transform)
# Load the test dataset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# set model to eval mode
model.eval()

# create lists to store predicted and true labels
predicted_labels = []
true_labels = []

# create list to store loss values
val_losses = []

# loop through validation set
for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    # get predictions
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        val_losses.append(loss.item())
        _, predicted = torch.max(outputs.logits.data, 1)

    # append predicted and true labels
    predicted_labels += predicted.cpu().numpy().tolist()
    true_labels += labels.cpu().numpy().tolist()

# calculate accuracy
accuracy = 100 * np.sum(np.array(predicted_labels) == np.array(true_labels)) / len(predicted_labels)
print('[TEST]   Loss: %.4f, Accuracy: %.2f %%' % (loss.item(), accuracy))

# calculate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# calculate average validation loss
avg_val_loss = np.mean(val_losses)

# Plot the train and validation accuracies and losses
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_acc, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# plot confusion matrix
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()



'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import transformers
from transformers import CvtForImageClassification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the pre-trained model
model_name = 'microsoft/cvt-13'
model = CvtForImageClassification.from_pretrained(model_name)

# Modify the number of output classes
num_classes = 4
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the dataset and define the optimizer and loss function
transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = torchvision.datasets.ImageFolder('.\datacvt', transform=transform)
val_dataset = torchvision.datasets.ImageFolder('.\datacvt', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Train the model
num_epochs = 1

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Epoch: %d, Loss: %.4f, Accuracy: %.2f %%' % (epoch + 1, loss.item(), 100 * correct / total))

test_dataset = torchvision.datasets.ImageFolder('.\datacvt', transform=transform)
# Load the test dataset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

# set model to eval mode
model.eval()

# create lists to store predicted and true labels
predicted_labels = []
true_labels = []

# create list to store loss values
val_losses = []

# loop through validation set
for inputs, labels in test_loader:
    # move data to device
    inputs = inputs.to(device)
    labels = labels.to(device)

    # get predictions
    with torch.no_grad():
        outputs
'''