"""
Test script to verify augmentation techniques on a single image
This helps you preview the effects before running on the entire dataset
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def test_augmentations(image_path):
    """Test all augmentation techniques on a single image"""

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply augmentations
    def image_logarithm(img):
        img_float = img.astype(np.float32)
        log_img = np.log(img_float + 1)
        log_img = (log_img / np.max(log_img)) * 255
        return log_img.astype(np.uint8)

    def image_exponential(img, gamma=0.5):
        normalized = img.astype(np.float32) / 255.0
        gamma_corrected = np.power(normalized, gamma)
        return (gamma_corrected * 255).astype(np.uint8)

    def mean_filter_convolution(img, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (
            kernel_size * kernel_size
        )
        if len(img.shape) == 3:
            filtered = np.zeros_like(img)
            for i in range(img.shape[2]):
                filtered[:, :, i] = cv2.filter2D(img[:, :, i], -1, kernel)
        else:
            filtered = cv2.filter2D(img, -1, kernel)
        return filtered.astype(np.uint8)

    # Apply transformations
    log_image = image_logarithm(image_rgb)
    exp_image = image_exponential(image_rgb)
    mean_filtered = mean_filter_convolution(image_rgb)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Dataset Augmentation Preview", fontsize=16)

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(log_image)
    axes[0, 1].set_title("Logarithmic Transformation\n(Enhances dark regions)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(exp_image)
    axes[1, 0].set_title("Exponential Transformation\n(Enhances bright regions)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(mean_filtered)
    axes[1, 1].set_title("Mean Filter (5x5)\n(Smooths noise)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

    print("Augmentation preview complete!")
    print("If the results look good, run the main augmentation script.")


if __name__ == "__main__":
    # Test with the first apple image (adjust path as needed)
    test_image_path = "dataset/apple/6-01-V1-B.png"

    if Path(test_image_path).exists():
        print(f"Testing augmentations on: {test_image_path}")
        test_augmentations(test_image_path)
    else:
        print(f"Test image not found: {test_image_path}")
        print(
            "Please adjust the test_image_path in the script to point to an existing image."
        )
