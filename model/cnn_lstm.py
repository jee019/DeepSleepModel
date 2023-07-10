import torch
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

device = torch.device(device)
print(device)

from scipy import io
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import os
import mat73
import torchvision.transforms as transforms
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
from torchinfo import summary
from sklearn.metrics import precision_score, recall_score, confusion_matrix

class CNN(nn.Module):
    def init(self):
        super(CNN, self).init()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(235200, 500)
        self.fc2 = nn.Linear(500, 50)
        self.lstm = nn.LSTM(input_size=50, hidden_size=20, num_layers=1, batch_first=True)
        self.fc3 = nn.Linear(20, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), -1, 50)  # reshape to [batch_size, seq_len, input_size] for LSTM

        # pass through LSTM layer
        h0 = torch.zeros(1, x.size(0), 20).to(x.device)
        c0 = torch.zeros(1, x.size(0), 20).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # take the final hidden state as output

        out = self.fc3(out)
        return out

cnn = CNN()
cnn = cnn.cuda()
criterion = torch.nn.CrossEntropyLoss()
summary(cnn, (50, 1, 141, 600))
optimizer = optim.SGD(cnn.parameters(), lr=0.01)

train_accu = []
eval_losses = []
eval_accu = []

for epoch in range(160):

    for i in range(1, 20):
        train_losses = []
        train_acc = []
        train_data = []
        val_data = []
        test_data = []

        train_stage1 = 0
        train_stage2 = 0
        train_stage3 = 0
        train_stage4 = 0

        val_stage1 = 0
        val_stage2 = 0
        val_stage3 = 0
        val_stage4 = 0

        test_stage1 = 0
        test_stage2 = 0
        test_stage3 = 0
        test_stage4 = 0

        trans = transforms.ToTensor()

        str1 = "/home/user1/다운로드/cwtnew/CWTData_s"
        str2 = str(i)
        str3 = str1 + str2 + ".mat"
        data = mat73.loadmat(str3)
        print(str3)
        cwtDataInfo = data.get("CWTData")
        Power = cwtDataInfo.get("Power")
        stg = cwtDataInfo.get("stg")

        timelen = int(Power.shape[1] / 600)
        start_time = (timelen - 1) * 600
        end_time = timelen * 600
        b = Power[0:140, start_time: end_time]
        train_Data = np.array(b, dtype=np.float32)

        for j in range(timelen):
            if stg[j - 1] == 3 or stg[j - 1] == 4:
                a = (trans(train_Data), int(0))
                train_data.append(a)
                train_stage1 += 1
            # N3
            elif stg[j - 1] == 5:
                a = (trans(train_Data), int(1))
                train_data.append(a)
                train_stage2 += 1
            # wake
            elif stg[j - 1] == 6:
                a = (trans(train_Data), int(2))
                train_data.append(a)
                train_stage3 += 1
            # REM
            elif stg[j - 1] == 7:
                a = (trans(train_Data), int(3))
                train_data.append(a)
                train_stage4 += 1

        print(len(train_data))
        random.shuffle(train_data)
        print('\ntrain 1: %d,   2: %d,  3: %d, 4: %d' % (train_stage1, train_stage2, train_stage3, train_stage4))
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=100, shuffle=True)

        cnn.train()  # 학습
        running_loss = 0
        correct = 0
        total = 0
        for index, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()  # 기울기 초기화
            output = cnn(data)
            loss = criterion(output, target)
            loss.backward()  # 역전파
            optimizer.step()

            running_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total += target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        accu = 100. * correct / total

        train_accu.append(accu)
        train_losses.append(train_loss)
        #print('\nEpoch : %d' % epoch)
        #print('Train Loss: %.3f | Accuracy: %.3f' % (train_loss, accu))

        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            data = mat73.loadmat("/home/user1/다운로드/cwtnew/CWTData_s20.mat")

            cwtDataInfo = data.get("CWTData")
            Power = cwtDataInfo.get("Power")
            stg = cwtDataInfo.get("stg")

            timelen = int(Power.shape[1] / 600)
            start_time = (timelen - 1) * 600
            end_time = timelen * 600
            b = Power[0:140, start_time: end_time]
            train_Data = np.array(b, dtype=np.float32)

            for j in range(timelen):
                if stg[j - 1] == 3 or stg[j - 1] == 4:
                    a = (trans(train_Data), int(0))
                    val_data.append(a)
                    val_stage1 += 1
                # N3
                elif stg[j - 1] == 5:
                    a = (trans(train_Data), int(1))
                    val_data.append(a)
                    val_stage2 += 1
                # wake
                elif stg[j - 1] == 6:
                    a = (trans(train_Data), int(2))
                    val_data.append(a)
                    val_stage3 += 1
                # REM
                elif stg[j - 1] == 7:
                    a = (trans(train_Data), int(3))
                    val_data.append(a)
                    val_stage4 += 1

            print(len(val_data))
            random.shuffle(val_data)
            print('val 1: %d,   2: %d,  3: %d, 4: %d' % (val_stage1, val_stage2, val_stage3, val_stage4))
            val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=100, shuffle=True)

            for index, (data, target) in enumerate(val_loader):
                data, target = data.cuda(), target.cuda()
                cnn.eval()

                output = cnn(data)

                loss = criterion(output, target)
                running_loss += loss.item()

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total += target.size(0)
                correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss = running_loss / len(val_loader.dataset)
            accu = 100. * correct / total
            eval_losses.append(val_loss)
            eval_accu.append(accu)


for epoch in range(1):
    cnn.eval()  # test case 학습 방지
    running_loss = 0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    data = mat73.loadmat("/home/user1/다운로드/cwtnew/CWTData_s21.mat")

    cwtDataInfo = data.get("CWTData")
    Power = cwtDataInfo.get("Power")
    stg = cwtDataInfo.get("stg")

    timelen = int(Power.shape[1] / 600)
    start_time = (timelen - 1) * 600
    end_time = timelen * 600
    b = Power[0:140, start_time: end_time]
    train_Data = np.array(b, dtype=np.float32)

    for j in range(timelen):
        if stg[j - 1] == 3 or stg[j - 1] == 4:
            a = (trans(train_Data), int(0))
            test_data.append(a)
            test_stage1 += 1
        # N3
        elif stg[j - 1] == 5:
            a = (trans(train_Data), int(1))
            test_data.append(a)
            test_stage2 += 1
        # wake
        elif stg[j - 1] == 6:
            a = (trans(train_Data), int(2))
            test_data.append(a)
            test_stage3 += 1
        # REM
        elif stg[j - 1] == 7:
            a = (trans(train_Data), int(3))
            test_data.append(a)
            test_stage4 += 1

    print(len(test_data))
    print('val 1: %d,   2: %d,  3: %d, 4: %d' % (test_stage1, test_stage2, test_stage3, test_stage4))
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=100, shuffle=True)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = cnn(data)

            loss = criterion(output, target)
            running_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total += target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)

            target = target.data.cpu().numpy()
            y_true.extend(target)

    test_loss = running_loss / len(test_loader.dataset)
    accu = 100. * correct / total
    print('Test Loss: %.3f | Accuracy: %.3f' % (test_loss, accu))

    print(np.shape(y_true))
    print(np.shape(y_pred))

    print(y_true[1])
    print(y_pred[1])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print(cm)

    accu_0 = 100. * (cm[0][0] / test_stage1)
    accu_1 = 100. * (cm[1][1] / test_stage2)
    accu_2 = 100. * (cm[2][2] / test_stage3)
    accu_3 = 100. * (cm[3][3] / test_stage4)

    print('label 0 accuracy: %.3f' % (accu_0))
    print('label 1 accuracy: %.3f' % (accu_1))
    print('label 2 accuracy: %.3f' % (accu_2))
    print('label 3 accuracy: %.3f' % (accu_3))

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="train")
plt.plot(eval_losses, label="test")
plt.xlabel('epoch')
plt.ylabel('losses')
plt.legend(['Train', 'Valid'])
plt.title('Train vs Valid Losses')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_accu, label="train")
plt.plot(eval_accu, label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(['Train', 'Valid'])
plt.title("Train vs Valid Accuracy")
plt.show()