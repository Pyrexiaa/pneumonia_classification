# # Download new_data.zip from google drive
# # Note: This dataset is already splitted based on the pneumonia type (bacteria and virus) and the normal images
# # The dataset is also resplit into train, test, and validation folders
# # https://drive.google.com/file/d/1h3WmTxwDAKy2eRw0SCXcNVFdreuquRQN/view?usp=sharing

# import gdown
# import zipfile

# url = 'https://drive.google.com/uc?id=1h3WmTxwDAKy2eRw0SCXcNVFdreuquRQN'
# output = 'new_data.zip'
# gdown.download(url, output, quiet=False)

# # Unzip the new_data.zip file
# with zipfile.ZipFile("new_data.zip", "r") as zip_ref:
#     zip_ref.extractall("dataset")

#############################################################
# TRAIN ALL 5 MODELS WITH 8 PREPROCESSING TECHNIQUES        #
# WITH PRETRAINED MODELS AND NEW DATASET                    #
#############################################################

## Import libraries
import os
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
import cv2

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from skimage.measure import label, regionprops
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score, precision_score, recall_score, accuracy_score
import seaborn as sns


def resize_image(image, target_size=(224, 224)):
    """
    Resizes an input image to the specified target size.
    
    Args:
        image (PIL Image or Torch Tensor): Input image to be resized.
        target_size (tuple): Desired output size (height, width).
    
    Returns:
        PIL Image: Resized image.
    """

    transform = transforms.Resize(target_size)
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    return transform(image)

def histogram_equalization(image):
    """
    Performs histogram equalization on a grayscale image and returns a tensor.
    
    Args:
        image (PIL Image or Torch Tensor): Input grayscale image.
    
    Returns:
        torch.Tensor: Equalized image as a tensor.
    """

    # Convert the image to a NumPy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image.convert('L'))  # Convert to grayscale
    elif isinstance(image, torch.Tensor):
        image = image.numpy()

    # Ensure the image is single-channel and np.uint8
    if image.ndim == 3 and image.shape[0] == 1:  # If shape is (1, H, W)
        image = image.squeeze(0)  # Remove the extra channel dimension
    elif image.ndim == 3 and image.shape[-1] == 1:  # If shape is (H, W, 1)
        image = image.squeeze(-1)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
        
    equalized_image = cv2.equalizeHist(image)
    equalized_pil_image = Image.fromarray(equalized_image)
    # equalized_tensor = torch.from_numpy(equalized_image).float().unsqueeze(0)  # Add channel dimension for grayscale
    return equalized_pil_image

def gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    Applies Gaussian blur to a grayscale image.
    
    Args:
        image (PIL Image or Torch Tensor): Input grayscale image.
        kernel_size (tuple): Size of the Gaussian kernel.
        sigma (float): Standard deviation for Gaussian kernel. 
                       If 0, it will be calculated based on the kernel size.
    
    Returns:
        torch.Tensor: Blurred image as a tensor.
    """

    # Convert to NumPy array if the image is a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        # Convert to NumPy if the input is a tensor
        image = image.numpy()
    
    # If the image has a channel dimension (1, H, W), squeeze it to (H, W)
    if image.ndim == 3 and image.shape[0] == 1:
        image = np.squeeze(image, axis=0)
    
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    blurred_tensor = torch.from_numpy(blurred_image).float().unsqueeze(0)  # Add channel dimension for grayscale

    return blurred_tensor

def bilateral_filter(image, diameter=5, sigma_color=75, sigma_space=75):
    """
    Applies a bilateral filter to a grayscale image.
    
    Args:
        image (PIL Image, NumPy array, or Torch Tensor): Input grayscale image.
        diameter (int): Diameter of each pixel neighborhood used in the filter.
        sigma_color (float): Filter sigma in the color space.
        sigma_space (float): Filter sigma in the coordinate space.
    
    Returns:
        torch.Tensor: Filtered image as a tensor.
    """
    # Convert to NumPy array if the image is a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    
    # If the image has a channel dimension (1, H, W), squeeze it to (H, W)
    if image.ndim == 3 and image.shape[0] == 1:
        image = np.squeeze(image, axis=0)

    if image.dtype != np.uint8:
        image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)
    
    # Apply bilateral filter using OpenCV
    filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    
    # Convert back to a PyTorch tensor
    filtered_tensor = torch.from_numpy(filtered_image).float().unsqueeze(0)  # Add back channel dimension for grayscale
    
    return filtered_tensor

def adaptive_masking(image, closing_kernel_size=(5, 5)):
    """
    Applies adaptive masking by removing the diaphragm from a grayscale image.
    
    Args:
        image (PIL Image, NumPy array, or Torch Tensor): Input grayscale image.
        closing_kernel_size (tuple): Size of the structuring element for morphological closing.
    
    Returns:
        torch.Tensor: Image with diaphragm removed as a tensor.
    """
    # Convert to NumPy array if the image is a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image.convert('L'))  # Ensure grayscale
    elif isinstance(image, torch.Tensor):
        # Convert to NumPy if the input is a tensor
        image = image.numpy()
    
    # If the image has a channel dimension (1, H, W), squeeze it to (H, W)
    if image.ndim == 3 and image.shape[0] == 1:
        image = np.squeeze(image, axis=0)
    
    # Step 1: Find max and min intensity values
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    
    # Step 2: Calculate threshold using the formula: threshold = min + 0.9 * (max - min)
    threshold_value = min_intensity + 0.9 * (max_intensity - min_intensity)
    
    # Step 3: Apply binary thresholding
    _, binary_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Step 4: Label connected regions and keep only the largest region
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)
    if not regions:
        print("No regions found in the binary mask.")
        return torch.from_numpy(image).float().unsqueeze(0)
    
    # Identify the largest connected region
    largest_region = max(regions, key=lambda r: r.area)
    
    # Create a mask with only the largest region filled
    diaphragm_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    diaphragm_mask[labeled_mask == largest_region.label] = 255
    
    # Step 5: Fill any holes in the diaphragm region
    diaphragm_mask = cv2.morphologyEx(diaphragm_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    # Step 6: Apply morphological closing to smooth mask (remove small holes)
    kernel = np.ones(closing_kernel_size, np.uint8)
    diaphragm_mask = cv2.morphologyEx(diaphragm_mask, cv2.MORPH_CLOSE, kernel)
    
    # Step 7: Bitwise operation to remove diaphragm from the source image
    result_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(diaphragm_mask))

    equalized_pil_image = Image.fromarray(result_image)
    
    return equalized_pil_image

preprocess_types = {
    "baseline": [resize_image],
    "histogram_equalization": [resize_image, histogram_equalization],
    "gaussian_blur": [resize_image, histogram_equalization, gaussian_blur],
    "bilateral_filer": [resize_image, histogram_equalization, bilateral_filter],
    "adaptive_masking": [resize_image, adaptive_masking],
    "adaptive_masking_equalized": [
        resize_image,
        adaptive_masking,
        histogram_equalization,
    ],
    "adaptive_masking_gaussian": [
        resize_image,
        adaptive_masking,
        histogram_equalization,
        gaussian_blur,
    ],
    "adaptive_masking_bilateral": [
        resize_image,
        adaptive_masking,
        histogram_equalization,
        bilateral_filter,
    ],
}

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """
    Trains and validates a model for a specified number of epochs.

    Parameters:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cuda' or 'cpu')
        num_epochs: Number of epochs

    Returns:
        history: Dictionary containing training and validation loss and accuracy
    """
    model.to(device)
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total

        # Logging
        print(f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f} - Validation Acc: {val_acc:.4f}")
        print()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return history

def test_model(model, test_loader, device):
    """
    Tests a model on a test set.

    Parameters:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to test on ('cuda' or 'cpu')

    Returns:
        y_true: True labels
        y_pred: Predicted labels
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            y_true += labels.tolist()
            y_pred += predictions.tolist()

        # compute accuracy
        accuracy = accuracy_score(y_true, y_pred)
    return accuracy

## Model pipeline
def model_pipelines(model, model_name, preprocess=None, root="dataset/new_data", save_path="models_pretrained"):
    # Result table
    results = np.array([['Preprocess', 'Test Accuracy']])
    os.makedirs(save_path, exist_ok=True)
    
    # Loop through the preprocess_types
    for key, value in preprocess_types.items():
        functions = preprocess_types[key]
        if preprocess is not None and key not in preprocess:
            continue
        print(f"\n===== {key} =====")
        transform = transforms.Compose(functions + [
            transforms.Lambda(lambda x: x.convert('L') if isinstance(x, Image.Image) else x), # convert to grayscale
            transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) else transforms.ToTensor()(x)), # convert to tensor (To ensure torch.Size([1, 224, 224]))
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert single channel to RGB (3 channels)
        ])
            
        train_data = datasets.ImageFolder(root=f'{root}/train', transform=transform)
        test_data = datasets.ImageFolder(root=f'{root}/test', transform=transform)
        val_data = datasets.ImageFolder(root=f'{root}/val', transform=transform)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        # Initialize the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        history = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20)
        
        # Evaluate the model
        accuracy = test_model(model, test_loader, device)
        torch.save(model, f"{save_path}/{model_name}_{key}.pth")
        results = np.append(results, [[key, accuracy]], axis=0)
        print(f"Test Accuracy: {accuracy}")
        print("\n")
        
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
        self.fc2 = nn.Linear(512, 3)  # Three classes: NORMAL, BACTERIA, VIRUS

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
        # 'VGG16': models.vgg16(pretrained=True),
        # 'ResNet50': models.resnet50(pretrained=True),
        # 'DenseNet161': models.densenet161(pretrained=True),
        # 'EfficientNetB1': models.efficientnet_b1(pretrained=True),
    }

    all_results = pd.DataFrame()
        
    for model_name, model in model_architectures.items():
        print(f"\n=========================\n{model_name}\n=========================")
        results = model_pipelines(model, model_name, preprocess=None) # include all 8 preprocess
        # results = model_pipelines(model, model_name, preprocess="baseline", save_path="models_baseline") # include 1 preprocess only
        results_df = pd.DataFrame(results[1:], columns=results[0])
        results_df['Model'] = model_name
        all_results = pd.concat([all_results, results_df], axis=0)

    original_sequence = all_results['Preprocess'].drop_duplicates().tolist()
    all_results = all_results.pivot(index='Preprocess', columns='Model', values='Test Accuracy').reindex(index=original_sequence)
    all_results.to_csv('result_CNN.csv')
    print("\n", all_results)
        
    