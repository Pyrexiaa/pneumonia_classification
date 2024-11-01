from .utils import *
import os
import shutil

def segregate_pneumonia_images(root_dir, output_dir):
    # Define dataset splits
    splits = ['train', 'test', 'validation']
    
    for split in splits:
        # Find the folder with pneumonia name
        pneumonia_folder = os.path.join(root_dir, split, 'pneumonia')
        
        # Check if pneumonia folder exists
        if not os.path.isdir(pneumonia_folder):
            print(f"Directory not found: {pneumonia_folder}")
            continue
        
        # Define output folders for BACTERIA and VIRUS
        bacteria_output_folder = os.path.join(output_dir, split, 'BACTERIA')
        virus_output_folder = os.path.join(output_dir, split, 'VIRUS')
        os.makedirs(bacteria_output_folder, exist_ok=True)
        os.makedirs(virus_output_folder, exist_ok=True)
        
        # List images in the current pneumonia folder
        image_files = [f for f in os.listdir(pneumonia_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in image_files:
            source_path = os.path.join(pneumonia_folder, image_file)
            
            # Check filename to determine if it's bacteria or virus
            if "bacteria" in image_file.lower():
                target_path = os.path.join(bacteria_output_folder, image_file)
            elif "virus" in image_file.lower():
                target_path = os.path.join(virus_output_folder, image_file)
            else:
                print(f"Skipping unknown file type: {image_file}")
                continue
            
            # Copy image to the respective output folder
            shutil.copy(source_path, target_path)
            print(f"Copied {image_file} to {target_path}")

def preprocess_and_save_sample_images(root_dir, output_dir, preprocess_func):
    """
    Reads images from a directory structure, applies preprocessing, 
    and saves one preprocessed image per class (normal, pneumonia) from each subfolder.
    
    Args:
        root_dir (str): Root directory containing `train`, `test`, and `validation` folders.
        output_dir (str): Directory to save sample preprocessed images.
        preprocess_func (function): Function to preprocess each image.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define dataset splits and classes
    splits = ['train', 'test', 'validation']
    classes = ['normal', 'pneumonia']
    
    # Process each split and class
    for split in splits:
        for c in classes:
            folder_path = os.path.join(root_dir, split, c)
            
            # Check if folder exists
            if not os.path.isdir(folder_path):
                print(f"Directory not found: {folder_path}")
                continue
            
            # List images in the current folder
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print(f"No images found in: {folder_path}")
                continue
            
            # Process only the first image as sample
            sample_image_path = os.path.join(folder_path, image_files[0])
            print(f"Processing sample image: {sample_image_path}")
            
            # Load image
            image = Image.open(sample_image_path).convert('L')  # Convert to grayscale
            
            # Apply preprocessing
            preprocessed_image = preprocess_func(image)
            
            # Convert to NumPy array for saving
            preprocessed_np = preprocessed_image.numpy().astype(np.uint8)
            
            # Define output path for saving
            output_path = os.path.join(output_dir, f"{split}_{c}_sample.png")
            Image.fromarray(preprocessed_np).save(output_path)
            
            print(f"Saved preprocessed sample to: {output_path}")

            # break after saving one sample to check
            break

def main():
    pass

if __name__ == '__main__':
    pass