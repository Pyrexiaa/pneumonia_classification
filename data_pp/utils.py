import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from skimage.measure import label, regionprops

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

# It will segment the lung areas to isolate regions of interest, allowing you to focus on the lung lobes and filter out other structures.
# This will return a mask that segments the lung from the rest of the image
def otsu_threshold(image):
    """
    Applies Otsu's thresholding on a grayscale image.
    
    Args:
        image (PIL Image, NumPy array, or Torch Tensor): Input grayscale image.
    
    Returns:
        torch.Tensor: Binary image after applying Otsu's thresholding.
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
    
    # Ensure the image is in np.uint8 format for OpenCV compatibility
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.dtype == np.float32 else image.astype(np.uint8)

    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to a PyTorch tensor
    binary_tensor = torch.from_numpy(binary_image).float().unsqueeze(0)  # Add back channel dimension for grayscale
    
    return binary_tensor

# It uses the Scharr operator to detect the presence and clarity of lobe boundaries or other anatomical edges. 
# Degraded or irregular edges might signal abnormalities related to pneumonia.
# This will return the original image with better edge emphasis.
def scharr_operator(image):
    """
    Applies Scharr's operator for edge detection on a grayscale image.
    
    Args:
        image (PIL Image, NumPy array, or Torch Tensor): Input grayscale image.
    
    Returns:
        torch.Tensor: Edge-detected image as a tensor.
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
    
    # Apply Scharr operator in the x and y directions
    scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    
    # Calculate the gradient magnitude
    scharr_edges = cv2.magnitude(scharr_x, scharr_y)
    scharr_edges = np.clip(scharr_edges, 0, 255).astype(np.uint8)  # Clip and convert to uint8
    
    # Convert back to a PyTorch tensor
    edges_tensor = torch.from_numpy(scharr_edges).float().unsqueeze(0)  # Add back channel dimension for grayscale
    
    return edges_tensor
