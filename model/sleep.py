import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import transformers
from transformers import CvtForImageClassification

# Load the pre-trained model
model_name = 'microsoft/cvt-13'
model = CvtForImageClassification.from_pretrained(model_name)

# Modify the number of output classes
num_classes = 1000
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# Load the dataset and define the optimizer and loss function
transform = transforms.Compose([
    #transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = torchvision.datasets.ImageFolder('/home/jeeyoung/다운로드/sleepImage/train', transform=transform)
val_dataset = torchvision.datasets.ImageFolder('/home/jeeyoung/다운로드/sleepImage/test', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Train the model
num_epochs = 1

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
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

    print('Epoch: %d, Loss: %.4f, Accuracy: %.2f %%' % (epoch+1, loss.item(), 100 * correct / total))
