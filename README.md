# Dataset Augmentation for Fruit Classification

This project contains scripts to augment your fruit dataset using three image processing techniques:

1. **Image Logarithm** - Enhances darker regions of images
2. **Image Exponential** - Enhances brighter regions of images
3. **Mean Filter Using Convolution** - Smooths noise and reduces detail

## Files Created

- `dataset_augmentation.py` - Main script to process entire dataset
- `test_augmentation.py` - Test script to preview effects on single image
- `requirements.txt` - Python dependencies
- `README.md` - This instruction file

## Setup Instructions

1. **Install Python dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

2. **Test the augmentation on a single image (recommended first step):**

   ```powershell
   python test_augmentation.py
   ```

3. **Run the full dataset augmentation:**
   ```powershell
   python dataset_augmentation.py
   ```

## What the Script Does

- Processes all images in your `dataset/` folder
- Creates a new `augmented_dataset_processed/` folder
- For each original image, creates 4 versions:
  - `*_original.png` - Copy of original image
  - `*_log.png` - Logarithmic transformation applied
  - `*_exp.png` - Exponential transformation applied
  - `*_mean.png` - Mean filter applied
- Maintains the same folder structure by fruit type

## Output Structure

```
augmented_dataset_processed/
├── apple/
│   ├── 6-01-V1-B_original.png
│   ├── 6-01-V1-B_log.png
│   ├── 6-01-V1-B_exp.png
│   ├── 6-01-V1-B_mean.png
│   └── ... (all other apple images with 4 versions each)
├── banana/
│   └── ... (all banana images with 4 versions each)
└── ... (all other fruit categories)
```

## Technical Details

### Image Logarithm

- Formula: `log(image + 1)`
- Effect: Compresses the dynamic range, making dark areas more visible
- Good for: Images with predominantly dark regions

### Image Exponential (Gamma Correction)

- Formula: `image^gamma` where gamma = 0.5
- Effect: Enhances bright regions, darkens mid-tones
- Good for: Images that are too bright or need contrast adjustment

### Mean Filter

- Uses a 5x5 kernel with convolution
- Effect: Smooths the image, reduces noise
- Good for: Removing noise while preserving overall structure

## Expected Results

After running the augmentation:

- **Original dataset**: ~200 images (20 per fruit category × 10 categories)
- **Augmented dataset**: ~800 images (4 versions × 200 original images)
- **Total increase**: 4x the original dataset size

## Troubleshooting

- If you get import errors, make sure you've installed the requirements
- If images don't load, check that your dataset folder structure matches the expected format
- The script includes detailed logging to help debug any issues

## Customization

You can modify the parameters in `dataset_augmentation.py`:

- `gamma` value in `image_exponential()` (default: 0.5)
- `kernel_size` in `mean_filter_convolution()` (default: 5)
- Add additional augmentation techniques as needed
