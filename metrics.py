import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data_pp.main import preprocess_types
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# load model
print("""
Model: DenseNet161 
Preprocess: adaptive_masking_equalized
Epoch: 20
      """)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('models/DenseNet161_adaptive_masking_equalized.pth', weights_only=False, map_location=device)

transform = transforms.Compose(preprocess_types['adaptive_masking_equalized'] + [
    transforms.Lambda(lambda x: x.convert('L') if isinstance(x, Image.Image) else x), # convert to grayscale
    transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) else transforms.ToTensor()(x)), # convert to tensor (To ensure torch.Size([1, 224, 224]))
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert single channel to RGB (3 channels)
])

test_data = datasets.ImageFolder(root='dataset/multiclass/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# print the class to index mapping and the number of samples in each class
print("\n======== Class to Index Mapping ========\n")
print(f"Test dataset size: {len(test_data)}\n") 
# table = pd.DataFrame({
#     'Index': range(len(test_data.classes)),
#     'Class Name': test_data.classes,
#     'Number of Samples': [test_data.targets.count(i) for i in range(len(test_data.classes))]
# })
table = pd.DataFrame({
    'Index': test_data.class_to_idx.values(),
    'Class Name': test_data.class_to_idx.keys(),
    'Number of Samples': [test_data.targets.count(i) for i in test_data.class_to_idx.values()]
})
print(table.to_string(index=False))

# get the true and predicted labels
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true += labels.tolist()
        y_pred += predicted.tolist()
        
# plot and save confusion matrix
cm = confusion_matrix(y_true, y_pred)
# replace class names using the class_to_idx mapping
class_names = [k for k, v in test_data.class_to_idx.items()]
plt.figure(figsize=(10, 7))
sns.set_theme(font_scale=1.4)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')

# print and save classification report
print("\n\n======== Classification Report ========\n")
print(classification_report(y_true, y_pred))
    
# calculate metrics
print("\n======== Metrics ========")
acc = accuracy_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

# print in tabular format
results = pd.DataFrame({
    'Accuracy': [acc],
    'Balanced Accuracy': [balanced_acc],
    'F1-Score': [f1],
    'Precision': [precision],
    'Recall': [recall]
}).T
print(results.to_string(header=False))