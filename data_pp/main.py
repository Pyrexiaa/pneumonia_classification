import os
import random
import shutil

from .utils import *

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
    "otsu": [
        resize_image,
        histogram_equalization,
        otsu_threshold,
    ],  # Act as additional feature for model training
    "scharr": [
        resize_image,
        histogram_equalization,
        scharr_operator,
    ],  # Act as additional feature for model training
}


# Split the pneumonia images into bacteria and virus infection
def segregate_pneumonia_images(root_dir, output_dir):
    splits = ["train", "test", "val"]
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        pneumonia_folder = os.path.join(root_dir, split, "pneumonia")
        if not os.path.isdir(pneumonia_folder):
            print(f"Directory not found: {pneumonia_folder}")
            continue
        # Define output folders for BACTERIA and VIRUS
        bacteria_output_folder = os.path.join(output_dir, split, "BACTERIA")
        virus_output_folder = os.path.join(output_dir, split, "VIRUS")
        os.makedirs(bacteria_output_folder, exist_ok=True)
        os.makedirs(virus_output_folder, exist_ok=True)

        image_files = [
            f
            for f in os.listdir(pneumonia_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for image_file in image_files:
            source_path = os.path.join(pneumonia_folder, image_file)

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

    # Randomly move 200 images from the train/virus folder to the val/virus folder
    train_virus_folder = os.path.join(output_dir, "train", "VIRUS")
    val_virus_folder = os.path.join(output_dir, "val", "VIRUS")
    os.makedirs(val_virus_folder, exist_ok=True)

    # Randomly move 192 images from the train/bacteria folder to the val/bacteria folder
    train_bacteria_folder = os.path.join(output_dir, "train", "BACTERIA")
    val_bacteria_folder = os.path.join(output_dir, "val", "BACTERIA")

    # Randomly move 192 images from the train/normal folder to the val/normal folder
    train_normal_folder = os.path.join(output_dir, "train", "NORMAL")
    val_normal_folder = os.path.join(output_dir, "val", "NORMAL")

    # Get all virus images in the train/virus folder
    virus_images = [
        f
        for f in os.listdir(train_virus_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    bacteria_images = [
        f
        for f in os.listdir(train_bacteria_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    normal_images = [
        f
        for f in os.listdir(train_normal_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Randomly select 200 virus images from training dataset
    selected_virus_images = random.sample(virus_images, min(200, len(virus_images)))
    # Randomly select 192 bacterial images from training dataset
    selected_bacteria_images = random.sample(
        bacteria_images, min(192, len(bacteria_images))
    )
    # Randomly select 192 normal images from training dataset
    selected_normal_images = random.sample(
        normal_images, min(192, len(normal_images))
    )

    for image_file in selected_virus_images:
        source_path = os.path.join(train_virus_folder, image_file)
        target_path = os.path.join(val_virus_folder, image_file)

        # Move the selected image to the val/virus folder
        shutil.move(source_path, target_path)
        print(f"Moved {image_file} from {train_virus_folder} to {val_virus_folder}")

    for image_file in selected_bacteria_images:
        source_path = os.path.join(train_bacteria_folder, image_file)
        target_path = os.path.join(val_bacteria_folder, image_file)

        # Move the selected image to the val/virus folder
        shutil.move(source_path, target_path)
        print(
            f"Moved {image_file} from {train_bacteria_folder} to {val_bacteria_folder}"
        )

    for image_file in selected_normal_images:
        source_path = os.path.join(train_normal_folder, image_file)
        target_path = os.path.join(val_normal_folder, image_file)

        # Move the selected image to the val/virus folder
        shutil.move(source_path, target_path)
        print(
            f"Moved {image_file} from {train_normal_folder} to {val_normal_folder}"
        )


def preprocess_and_save_sample_images(root_dir, output_dir):
    """Reads images from a directory structure, applies preprocessing,
    and saves one preprocessed image per class (normal, pneumonia) from each subfolder.

    Args:
        root_dir (str): Root directory containing `train`, `test`, and `validation` folders.
        output_dir (str): Directory to save sample preprocessed images.
        preprocess_func (function): Function to preprocess each image.

    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    splits = ["train", "test", "val"]
    # If we want to do binary classification, can use the below class
    # classes = ['normal', 'pneumonia']
    classes = ["normal", "bacteria", "virus"]

    # Process each split and class
    for split in splits:
        for c in classes:
            folder_path = os.path.join(root_dir, split, c)

            # Check if folder exists
            if not os.path.isdir(folder_path):
                print(f"Directory not found: {folder_path}")
                continue

            # List images in the current folder
            image_files = [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            if not image_files:
                print(f"No images found in: {folder_path}")
                continue

            # Process only the first image as sample
            sample_image_path = os.path.join(folder_path, image_files[0])
            print(f"Processing sample image: {sample_image_path}")

            # Load image
            image = Image.open(sample_image_path).convert("L")  # Convert to grayscale
            current_image = image
            # Apply preprocessing
            for key, value in preprocess_types.items():
                functions = preprocess_types[key]
                for func in functions:
                    current_image = func(current_image)

                preprocessed_image = current_image

                if isinstance(preprocessed_image, Image.Image):
                    # If it's a PIL Image, ensure it is grayscale and convert to a numpy array
                    if preprocessed_image.mode != "L":
                        preprocessed_image = preprocessed_image.convert("L")
                    preprocessed_np = np.array(preprocessed_image)
                elif isinstance(preprocessed_image, torch.Tensor):
                    if (
                        preprocessed_image.dim() == 3
                        and preprocessed_image.shape[0] == 1
                    ):
                        preprocessed_image = preprocessed_image.squeeze(
                            0,
                        )  # Remove channel dimension if single-channel
                    preprocessed_np = preprocessed_image.numpy().astype(np.uint8)
                else:
                    raise TypeError(
                        "Preprocessed image is neither a PIL Image nor a torch.Tensor",
                    )

                output_folder = os.path.join(output_dir, key)
                output_path = os.path.join(output_folder, f"{split}_{c}_sample.png")
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                Image.fromarray(preprocessed_np).save(output_path)

                print(f"Saved preprocessed sample to: {output_path}")

            continue
        continue


def main(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # splits = ['train', 'test', 'val'] DO NOT AUGMENT THE TEST SET
    splits = ["train", "val"]
    # If we want to do binary classification, can use the below class
    classes = ["normal", "bacteria", "virus"]

    # Process each split and class
    for split in splits:
        for c in classes:
            folder_path = os.path.join(root_dir, split, c)

            # Check if folder exists
            if not os.path.isdir(folder_path):
                print(f"Directory not found: {folder_path}")
                continue

            # List images in the current folder
            image_files = [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            if not image_files:
                print(f"No images found in: {folder_path}")
                continue

            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                print(f"Processing image: {image_path}")

                # Load image
                image = Image.open(image_path).convert("L")  # Convert to grayscale
                current_image = image
                # Apply preprocessing
                for key, value in preprocess_types.items():
                    functions = preprocess_types[key]
                    for func in functions:
                        current_image = func(current_image)

                    preprocessed_image = current_image

                    if isinstance(preprocessed_image, Image.Image):
                        # If it's a PIL Image, ensure it is grayscale and convert to a numpy array
                        if preprocessed_image.mode != "L":
                            preprocessed_image = preprocessed_image.convert("L")
                        preprocessed_np = np.array(preprocessed_image)
                    elif isinstance(preprocessed_image, torch.Tensor):
                        if (
                            preprocessed_image.dim() == 3
                            and preprocessed_image.shape[0] == 1
                        ):
                            preprocessed_image = preprocessed_image.squeeze(
                                0,
                            )  # Remove channel dimension if single-channel
                        preprocessed_np = preprocessed_image.numpy().astype(np.uint8)
                    else:
                        raise TypeError(
                            "Preprocessed image is neither a PIL Image nor a torch.Tensor",
                        )

                    output_folder = os.path.join(output_dir, key, split, c)
                    output_path = os.path.join(output_folder, image_file)
                    os.makedirs(output_folder, exist_ok=True)

                    # Save the preprocessed image
                    Image.fromarray(preprocessed_np).save(output_path)
                    print(f"Saved preprocessed image to: {output_path}")


if __name__ == "__main__":
    # FIRST: Segregate the pneumonia images into 2 classes
    root_dir = "dataset"
    output_dir = "dataset"
    segregate_pneumonia_images(root_dir, output_dir)

    # SECOND: Debugging purpose to get sample image for every preprocessing function combinations
    # root_dir = 'dataset'
    # output_dir = 'preprocessed_sample_images'
    # preprocess_and_save_sample_images(root_dir, output_dir)

    # THIRD: Prepare all of the preprocessed images for training purpose
    # root_dir = 'dataset'
    # output_dir = 'modified_dataset'
    # main(root_dir, output_dir)
