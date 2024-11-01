# pneumonia_classification

WID3011 Deep Learning Group Assignment

# Dataset source

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# Setup

1. Download the dataset from the above website
2. Extract the train, test, validation folders from the zip file
3. Create a folder named 'dataset' and insert the folders into the 'dataset' folder

# Data Preprocessing

## Steps:
1. Requirements
    - Please install the requirements needed according to the **requirements.txt** for data preprocessing purpose.

2. Preprocessing Functions
    - Open **utils.py** to review the preprocessing functions applied to the images, including resizing, histogram equalization, Gaussian blur, bilateral filtering, adaptive masking, Otsu thresholding, and the Scharr operator for edge detection.

3. Dataset Segmentation and Image Sampling
    - In **main.py**, you’ll find functions for organizing the dataset into train, test, and val folders, each containing images categorized as normal lung, bacterial infection, and viral infection. There is also a debug function to sample images from the dataset, enabling you to inspect the images after each preprocessing step.

4. Creating the Final Dataset for Training
    - The main function in **main.py** saves all preprocessed images into a modified_dataset folder, which will be used for training. Each step is labeled as "first," "second," and "third" to guide the processing order. You can choose to run all steps sequentially at once if preferred.

## Reference:
- Pre-processing Methods in Chest X-Ray Image Classification: https://pmc.ncbi.nlm.nih.gov/articles/PMC8982897/pdf/pone.0265949.pdf
- Chest X-Ray Image Preprocessing for Disease Classification: https://www.sciencedirect.com/science/article/pii/S1877050921015556