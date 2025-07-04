"""
Dataset Augmentation Script for Fruit Classification Project

This script applies three augmentation techniques to all images in the dataset:
1. Image Logarithm - Enhances darker regions
2. Image Exponential - Enhances brighter regions
3. Mean Filter Using Convolution - Smooths noise

Author: Generated for PDI Project
Date: July 2025
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetAugmenter:
    def __init__(self, input_dir, output_dir):
        """
        Initialize the dataset augmenter

        Args:
            input_dir (str): Path to the original dataset directory
            output_dir (str): Path to save augmented images
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def image_logarithm(self, image):
        """
        Apply logarithmic transformation to enhance darker regions

        Args:
            image (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Logarithmically transformed image
        """
        # Convert to float to avoid overflow
        image_float = image.astype(np.float32)

        # Add 1 to avoid log(0) and apply logarithm
        log_image = np.log(image_float + 1)

        # Normalize to 0-255 range
        log_image = (log_image / np.max(log_image)) * 255

        return log_image.astype(np.uint8)

    def image_exponential(self, image, gamma=0.5):
        """
        Apply exponential (gamma) transformation to enhance brighter regions

        Args:
            image (numpy.ndarray): Input image
            gamma (float): Gamma value for transformation (< 1 enhances bright regions)

        Returns:
            numpy.ndarray: Exponentially transformed image
        """
        # Normalize image to 0-1 range
        normalized = image.astype(np.float32) / 255.0

        # Apply gamma transformation
        gamma_corrected = np.power(normalized, gamma)

        # Convert back to 0-255 range
        exp_image = (gamma_corrected * 255).astype(np.uint8)

        return exp_image

    def mean_filter_convolution(self, image, kernel_size=5):
        """
        Apply mean filter using convolution to smooth the image

        Args:
            image (numpy.ndarray): Input image
            kernel_size (int): Size of the mean filter kernel

        Returns:
            numpy.ndarray: Filtered image
        """
        # Create mean filter kernel
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (
            kernel_size * kernel_size
        )

        # Apply convolution for each channel if image is colored
        if len(image.shape) == 3:
            filtered_image = np.zeros_like(image)
            for i in range(image.shape[2]):
                filtered_image[:, :, i] = cv2.filter2D(image[:, :, i], -1, kernel)
        else:
            filtered_image = cv2.filter2D(image, -1, kernel)

        return filtered_image.astype(np.uint8)

    def process_single_image(self, image_path, output_folder):
        """
        Process a single image with all augmentation techniques

        Args:
            image_path (Path): Path to the input image
            output_folder (Path): Output folder for augmented images
        """
        try:
            # Read the original image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return

            # Get filename without extension
            filename_stem = image_path.stem
            filename_ext = image_path.suffix

            # Apply augmentations
            log_image = self.image_logarithm(image)
            exp_image = self.image_exponential(image)
            mean_filtered = self.mean_filter_convolution(image)

            # Save augmented images with descriptive suffixes
            log_path = output_folder / f"{filename_stem}_log{filename_ext}"
            exp_path = output_folder / f"{filename_stem}_exp{filename_ext}"
            mean_path = output_folder / f"{filename_stem}_mean{filename_ext}"

            cv2.imwrite(str(log_path), log_image)
            cv2.imwrite(str(exp_path), exp_image)
            cv2.imwrite(str(mean_path), mean_filtered)

            # Original images are not saved - only augmented versions
            logger.debug(f"Processed: {image_path.name}")

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")

    def augment_dataset(self):
        """
        Process the entire dataset applying all augmentation techniques
        """
        logger.info(
            f"Starting dataset augmentation from {self.input_dir} to {self.output_dir}"
        )

        # Get all fruit categories (subdirectories)
        fruit_categories = [d for d in self.input_dir.iterdir() if d.is_dir()]

        total_images = 0
        for category in fruit_categories:
            image_files = (
                list(category.glob("*.png"))
                + list(category.glob("*.jpg"))
                + list(category.glob("*.jpeg"))
            )
            total_images += len(image_files)

        logger.info(
            f"Found {len(fruit_categories)} categories with {total_images} total images"
        )

        # Process each category
        with tqdm(total=total_images, desc="Augmenting images") as pbar:
            for category_folder in fruit_categories:
                category_name = category_folder.name
                logger.info(f"Processing category: {category_name}")

                # Create output folder for this category
                output_category_folder = self.output_dir / category_name
                output_category_folder.mkdir(exist_ok=True)

                # Get all image files in this category
                image_files = (
                    list(category_folder.glob("*.png"))
                    + list(category_folder.glob("*.jpg"))
                    + list(category_folder.glob("*.jpeg"))
                )

                # Process each image
                for image_path in image_files:
                    self.process_single_image(image_path, output_category_folder)
                    pbar.update(1)

        logger.info("Dataset augmentation completed successfully!")

        # Print summary statistics
        self.print_summary()

    def print_summary(self):
        """Print summary of augmentation results"""
        logger.info("\n" + "=" * 50)
        logger.info("AUGMENTATION SUMMARY")
        logger.info("=" * 50)

        for category_folder in self.output_dir.iterdir():
            if category_folder.is_dir():
                image_count = len(
                    list(category_folder.glob("*.png"))
                    + list(category_folder.glob("*.jpg"))
                    + list(category_folder.glob("*.jpeg"))
                )
                logger.info(f"{category_folder.name}: {image_count} images")

        logger.info("=" * 50)


def main():
    """Main function to run the augmentation process"""

    # Define paths
    input_dataset_dir = "dataset"
    output_dataset_dir = "augmented_dataset_processed"

    # Check if input directory exists
    if not Path(input_dataset_dir).exists():
        logger.error(f"Input directory '{input_dataset_dir}' does not exist!")
        logger.info(
            "Please make sure you're running this script from the project root directory."
        )
        return

    # Create augmenter instance
    augmenter = DatasetAugmenter(input_dataset_dir, output_dataset_dir)

    # Run augmentation
    try:
        augmenter.augment_dataset()
        logger.info(
            f"\nAugmentation complete! Check the '{output_dataset_dir}' folder for results."
        )
        logger.info("Each original image now has 3 augmented versions:")
        logger.info("  - *_log.png: Logarithmic transformation")
        logger.info("  - *_exp.png: Exponential transformation")
        logger.info("  - *_mean.png: Mean filter applied")

    except Exception as e:
        logger.error(f"Augmentation failed: {str(e)}")


if __name__ == "__main__":
    main()
