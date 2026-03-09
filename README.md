# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model. 

## DESIGN STEPS
### STEP 1:
Import required libraries.Then dataset is loaded and define the training and testing dataset.

### STEP 2:
initialize the model,loss function,optimizer. CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

### STEP 3:
Train the model with training dataset.

### STEP 4:
Evaluate the model with testing dataset.

### STEP 5:
Make Predictions on New Data.

## PROGRAM
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for pre-trained model input
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
])
!unzip -qq ./chip_data.zip -d data
dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()
show_sample_images(train_dataset)
print(f"Total number of training samples: {len(train_dataset)}")

# Get the shape of the first image in the dataset
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")
print(f"Total number of testing samples: {len(test_dataset)}")

# Get the shape of the first image in the test dataset
first_test_image, test_label = test_dataset[0]
print(f"Shape of the first test image: {first_test_image.shape}")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = models.vgg19(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary(model, input_size=(3, 224, 224))
for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor layers
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
def train_model(model, train_loader, test_loader, num_epochs=10):
    train_losses = []
    val_losses = []

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)   # only once

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))

        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot loss graph
    print("Name: Nandhini M")
    print("Register Number: 212224040211")

    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', marker='s')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train_model(model, train_loader, test_loader, num_epochs=10)
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Apply sigmoid for binary classification
            probs = torch.sigmoid(outputs)

            # Convert probability to class (0 or 1)
            predicted = (probs > 0.5).int().squeeze()

            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("Name: Nandhini M")
    print("Register Number: 212224040211")

    plt.figure(figsize=(12,10))

    sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes,
            annot_kws={"size":16})

    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title('Confusion Matrix', fontsize=18)

    plt.show()
test_model(model, test_loader)
def predict_image(model, image_index, dataset):
    model.eval()

    image, label = dataset[image_index]

    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)

        output = model(image_tensor)

        # Apply sigmoid for binary classification
        prob = torch.sigmoid(output.squeeze())

        # Convert probability to class (0 or 1)
        predicted = (prob > 0.5).int().item()

    class_names = dataset.classes

    # Display image
    image_to_display = transforms.ToPILImage()(image)

    plt.figure(figsize=(4,4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()

    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')
predict_image(model, image_index=55, dataset=test_dataset)
predict_image(model, image_index=25, dataset=test_dataset)
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

<img width="862" height="768" alt="image" src="https://github.com/user-attachments/assets/bc1e89ce-5cb9-4532-a0cd-4568b45d2c21" />

### Confusion Matrix

<img width="891" height="772" alt="image" src="https://github.com/user-attachments/assets/dd0d588f-5730-46ef-974c-f6df63944f7b" />

### Classification Report

<img width="338" height="391" alt="image" src="https://github.com/user-attachments/assets/6a002545-4c9a-46cd-8341-777c709a547a" />

### New Sample Prediction

<img width="331" height="390" alt="image" src="https://github.com/user-attachments/assets/e61052f9-2a7a-485f-b0c2-700f479f3ebd" />

## RESULT
Thus, the transfer Learning for classification using VGG-19 architecture has succesfully implemented.

