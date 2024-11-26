import os
import shutil

## Move all files in the PNEUMONIA folder to the respective BACTERIA and VIRUS folders
# Paths
base_dir = "dataset/multiclass"
splits = ['train', 'val', 'test']
categories = ["bacteria", "virus"]

for split in splits:
    pneumonia_path = os.path.join(base_dir, split, "PNEUMONIA")
    
    # list all files in the pneumonia folder
    files = os.listdir(pneumonia_path)
    os.makedirs(os.path.join(base_dir, split, "BACTERIA"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, "VIRUS"), exist_ok=True)
    
    for file in files:
        if "_bacteria_" in file:
            shutil.move(os.path.join(pneumonia_path, file), os.path.join(base_dir, split, "BACTERIA", file))
        elif "_virus_" in file:
            shutil.move(os.path.join(pneumonia_path, file), os.path.join(base_dir, split, "VIRUS", file))
            
    # Remove the now-empty PNEUMONIA folder
    shutil.rmtree(pneumonia_path)
