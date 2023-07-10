import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
# tensorboard --logdir=logs/

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 데이터셋 경로
data_path = './train'

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class SleepDataset(Dataset):
    def __init__(self, data_path, transform=transform):
        self.data_path = data_path
        self.persons = os.listdir(self.data_path)
        self.transform = transform

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx):
        person = self.persons[idx]
        person_path = os.path.join(self.data_path, person)
        stages = os.listdir(person_path)

        # 각 수면 단계 폴더에서 이미지 읽어오기
        images = []
        labels = []
        for stage in stages:
            stage_path = os.path.join(person_path, stage)
            files = os.listdir(stage_path)
            for file in files:
                file_path = os.path.join(stage_path, file)
                img = Image.open(file_path)
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                img = img.to(device)
                images.append(img)
                labels.append(int(stage))

        labels = torch.tensor(labels).to(device)

        return images, labels


from sklearn.model_selection import train_test_split

# 전체 데이터셋
dataset = SleepDataset(data_path)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

train_ds, val_test_ds = train_test_split(dataset, test_size=0.2, random_state=42)
val_ds, test_ds = train_test_split(val_test_ds, test_size=0.5, random_state=42)

id2label = {3: 'N3', 4: 'N2', 5: 'N1', 6: 'WAKE', 7: 'REM'}
label2id = {'N3': 3, 'N2': 4, 'N1': 5, 'WAKE': 6, "REM": 7}

from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
    [
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

_val_transforms = Compose(
    [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ]
)


def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples


def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples


# Set the transforms
#train_ds.set_transform(train_transforms)
#val_ds.set_transform(val_transforms)
#test_ds.set_transform(val_transforms)

print(train_ds[:2])

from torch.utils.data import DataLoader
import torch


#def collate_fn(examples):
#    pixel_values = torch.stack([example["pixel_values"] for example in examples])
#    labels = torch.tensor([example["label"] for example in examples])
#    return {"pixel_values": pixel_values, "labels": labels}


#train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)

#batch = next(iter(train_dataloader))
#for k, v in batch.items():
#    if isinstance(v, torch.Tensor):
#        print(k, v.shape)

from transformers import ViTForImageClassification
#model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',id2label=id2label,label2id=label2id)
model_name = 'google/vit-base-patch16-224-in21k'
model = ViTForImageClassification.from_pretrained(model_name,
                                                  id2label=id2label,
                                                  label2id=label2id, output_hidden_states=False)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
from transformers import TrainingArguments, Trainer

metric_name = "accuracy"

args = TrainingArguments(
    f"test-cifar-10",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
)

from sklearn.metrics import accuracy_score
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))


import torch

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    #data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()

outputs = trainer.predict(test_ds)

print(outputs.metrics)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

labels = train_ds.features['label'].names
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)
