"""
Script to remove all *_original.png files from the augmented dataset
This keeps only the processed versions (log, exp, mean)
"""

import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def remove_original_files(dataset_dir):
    """
    Remove all *_original.png files from the augmented dataset

    Args:
        dataset_dir (str): Path to the augmented dataset directory
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        logger.error(f"Dataset directory '{dataset_dir}' does not exist!")
        return

    logger.info(f"Scanning '{dataset_dir}' for *_original.png files...")

    # Find all *_original.png files recursively
    original_files = list(dataset_path.glob("**/*_original.png"))

    if not original_files:
        logger.info("No *_original.png files found.")
        return

    logger.info(f"Found {len(original_files)} original files to remove.")

    # Remove each file
    removed_count = 0
    for file_path in original_files:
        try:
            file_path.unlink()  # Delete the file
            logger.debug(f"Removed: {file_path}")
            removed_count += 1
        except Exception as e:
            logger.error(f"Failed to remove {file_path}: {str(e)}")

    logger.info(f"Successfully removed {removed_count} original files.")

    # Print summary by category
    logger.info("\nRemaining files by category:")
    for category_folder in dataset_path.iterdir():
        if category_folder.is_dir():
            remaining_files = len(list(category_folder.glob("*.png")))
            logger.info(f"  {category_folder.name}: {remaining_files} files")


def main():
    """Main function"""
    dataset_dir = "augmented_dataset_processed"

    logger.info("=== Removing Original Files from Augmented Dataset ===")
    remove_original_files(dataset_dir)
    logger.info("=== Cleanup Complete ===")


if __name__ == "__main__":
    main()
