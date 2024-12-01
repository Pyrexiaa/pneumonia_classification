## Import libraries
import os
import shutil
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_pp.main import preprocess_types
import numpy as np
import pandas as pd
from PIL import Image


## Train the model
def train_model(model, train_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    
## Test the model    
def test_model(model, test_loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {100 * correct / total}%")
        return 100 * correct / total


## Model pipeline
def model_pipelines(model, model_name, preprocess=None, save_path="models"):
    # Result table
    results = np.array([['Preprocess', 'Test Accuracy']])
    os.makedirs(save_path, exist_ok=True)

    # Loop through the preprocess_types
    for key, value in preprocess_types.items():
        if preprocess is not None and key not in preprocess:
            continue
        functions = preprocess_types[key]
        print(f"\n===== {key} =====")
        transform = transforms.Compose(functions + [
            transforms.Lambda(lambda x: x.convert('L') if isinstance(x, Image.Image) else x), # convert to grayscale
            transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) else transforms.ToTensor()(x)), # convert to tensor (To ensure torch.Size([1, 224, 224]))
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert single channel to RGB (3 channels)
        ])
        
        # 1. Load datasets and transform them
        train_data = datasets.ImageFolder(root='dataset/binary/train', transform=transform)
        test_data = datasets.ImageFolder(root='dataset/binary/test', transform=transform)
        # train_data = datasets.ImageFolder(root='dataset/multiclass/train', transform=transform)
        # test_data = datasets.ImageFolder(root='dataset/multiclass/test', transform=transform)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # 2. Initialize the model    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 3. Train the model
        train_model(model, train_loader, device, epochs=20)
        
        ## 4. Evaluate the model
        accuracy = test_model(model, test_loader, device)
        torch.save(model, f"{save_path}/{model_name}_{key}.pth")
        results = np.append(results, [[key, accuracy]], axis=0)
    return results
   
   
## Define the CNN model
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout4 = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(512, 3)  # Three classes: NORMAL, BACTERIA, VIRUS
        self.fc2 = nn.Linear(512, 2)  # Two classes: NORMAL, PNEUMONIA

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout3(x)
        
        x = x.view(-1, 128 * 28 * 28)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x
     
              
## Main  
if __name__ == '__main__':
    model_architectures = {
        'CNN': PneumoniaCNN(),
        'VGG16': models.vgg16(),
        'ResNet50': models.resnet50(),
        'DenseNet161': models.densenet161(),
        'EfficientNetB1': models.efficientnet_b1(),
        # 'InceptionV3': models.inception_v3() # input size: (299, 299), output: InceptionOutputs not tensor
    }

    all_results = pd.DataFrame()
    for model_name, model in model_architectures.items():
        print(f"\n=========================\n{model_name}\n=========================")
        results = model_pipelines(model, model_name, preprocess=None, save_path="models_binary") # include all 10 preprocess
        # results = model_pipelines(model, model_name, preprocess="baseline", save_path="models_baseline") # include 1 preprocess only
        results_df = pd.DataFrame(results[1:], columns=results[0])
        results_df['Model'] = model_name
        all_results = pd.concat([all_results, results_df], axis=0)

    original_sequence = all_results['Preprocess'].drop_duplicates().tolist()
    all_results = all_results.pivot(index='Preprocess', columns='Model', values='Test Accuracy').reindex(index=original_sequence)
    all_results.to_csv('results_binary.csv')
    print("\n", all_results)
    