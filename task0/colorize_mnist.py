import numpy as np
import os
import sys
from pathlib import Path
import cv2

# Dominant colors for each digit (0-9)
DOMINANT_COLORS = {
    0: np.array([255, 0, 0], dtype=np.uint8),      # Red
    1: np.array([0, 128, 0], dtype=np.uint8),      # Green
    2: np.array([0, 0, 255], dtype=np.uint8),      # Blue
    3: np.array([255, 255, 0], dtype=np.uint8),    # Yellow
    4: np.array([0, 255, 255], dtype=np.uint8),    # Cyan
    5: np.array([255, 20, 147], dtype=np.uint8),   # neon pink
    6: np.array([255, 165, 0], dtype=np.uint8),    # Orange
    7: np.array([75, 0, 130], dtype=np.uint8),     # indigo
    8: np.array([255, 192, 203], dtype=np.uint8),  # light Pink
    9: np.array([128, 128, 0], dtype=np.uint8)     # olive green
}

def get_random_color():
    """Generate a random RGB color."""
    return np.random.randint(0, 256, size=3, dtype=np.uint8)

def add_background_noise(image, noise_level=15):
    """
    Add random background noise to the image.
    
    Args:
        image: RGB image (H, W, 3)
        noise_level: Standard deviation of Gaussian noise (default 15)
    
    Returns:
        Image with background noise added
    """
    noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
    noisy_image = image.astype(np.int16) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_foreground_strokes(image, num_strokes=None, stroke_color=None):
    """
    Add random strokes/lines to the image foreground.
    
    Args:
        image: RGB image (H, W, 3)
        num_strokes: Number of strokes to add (random 1-3 if None)
        stroke_color: RGB color for strokes (random if None)
        thickness_range: Tuple of (min, max) thickness
    
    Returns:
        Image with foreground strokes added
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    # Random number of strokes i
    # f not specified
    if num_strokes is None:
        num_strokes = np.random.randint(1, 3)  # 1-2 strokes
    
    for _ in range(num_strokes):
        # Random start and end points
        pt1 = (np.random.randint(0, w), np.random.randint(0, h))
        pt2 = (np.random.randint(0, w), np.random.randint(0, h))
        
        # Random color if not specified
        if stroke_color is None:
            color = tuple(map(int, np.random.randint(0, 256, 3)))
        else:
            color = tuple(map(int, stroke_color))
        
        # Random thickness
        thickness = 1
        
        # Draw line
        cv2.line(result, pt1, pt2, color, thickness)
    
    return result

def apply_color_transformation(image, label, use_dominant=True, add_noise=True, add_strokes=True):
    """
    Apply color transformation to a grayscale image.
    
    Args:
        image: Grayscale image (H, W, 3) where all channels are the same
        label: Digit label (0-9)
        use_dominant: If True, use dominant color; if False, use random color
        add_noise: If True, add background noise
        add_strokes: If True, add foreground strokes
    
    Returns:
        Colored image (H, W, 3)
    """
    # Get the grayscale values (take first channel since all are the same)
    grayscale = image[:, :, 0].astype(np.float32)
    
    # Choose transformation color
    if use_dominant:
        transform_color = DOMINANT_COLORS[label].astype(np.float32)
    else:
        transform_color = get_random_color().astype(np.float32)
    
    # Create colored image
    # Pixel whiteness ratio: grayscale / 255
    # More white (255) -> full transformation color
    # More black (0) -> stays black (0, 0, 0)
    h, w = grayscale.shape
    colored_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply color proportional to whiteness
    whiteness_ratio = grayscale / 255.0
    
    for c in range(3):
        colored_image[:, :, c] = (whiteness_ratio * transform_color[c]).astype(np.uint8)
    
    # Add background noise (70% probability)
    
    if add_noise and np.random.random() < 0.7:
        noise_level = np.random.randint(15,25)  # Random noise level
        colored_image = add_background_noise(colored_image, noise_level)
    
    # Add foreground strokes (50% probability)
    if add_strokes and np.random.random() < 0.5:
        colored_image = add_foreground_strokes(colored_image)
    
    return colored_image

def process_npz_file(input_path, output_path, bias_probability=0.95):
    """
    Process a single .npz file and apply color transformations.
    
    Args:
        input_path: Path to input .npz file
        output_path: Path to output .npz file
        bias_probability: Probability of using dominant color (default 0.95)
    """
    print(f"Processing {input_path.name}...")
    
    # Load data
    data = np.load(input_path)
    images = data['images']
    labels = data['labels']
    
    num_images = len(images)
    colored_images = np.zeros_like(images, dtype=np.uint8)
    
    # Process each image
    for i in range(num_images):
        image = images[i]
        label = labels[i]
        
        # Generate random number to decide color choice
        rand_val = np.random.random()
        use_dominant = rand_val < bias_probability
        
        # Apply color transformation
        colored_images[i] = apply_color_transformation(image, label, use_dominant)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{num_images} images...")
    
    # Save colored dataset
    np.savez(output_path, images=colored_images, labels=labels)
    print(f"Saved to {output_path.name}")
    print(f"  Shape: {colored_images.shape}, Labels: {labels.shape}")

def main():
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python colorize_mnist.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    
    # Validate input folder
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist!")
        sys.exit(1)
    
    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory!")
        sys.exit(1)
    
    # Check if output folder exists
    if output_folder.exists():
        print(f"Error: Output folder '{output_folder}' already exists!")
        print("Please choose a different output folder or delete the existing one.")
        sys.exit(1)
    
    # Create output folder
    output_folder.mkdir(parents=True)
    print(f"Created output folder: {output_folder}")
    
    # Find all .npz files in input folder
    npz_files = list(input_folder.glob("*.npz"))
    
    if not npz_files:
        print(f"Warning: No .npz files found in {input_folder}")
        sys.exit(1)
    
    print(f"\nFound {len(npz_files)} .npz file(s) to process")
    print(f"\nColor mapping:")
    for digit, color in DOMINANT_COLORS.items():
        print(f"  Digit {digit}: RGB{tuple(color)}")
    print(f"\nBias: 95% dominant color, 5% random color")
    print(f"Augmentation: 70% background noise, 50% foreground strokes")
    print()
    
    # Determine suffix based on output folder name
    folder_name = output_folder.name
    suffix = ""
    
    # Extract suffix from folder name (everything from first underscore)
    if "_" in folder_name:
        idx = folder_name.index("_")
        suffix = folder_name[idx:]
    
    if suffix:
        print(f"Adding '{suffix}' suffix to output filenames\n")
    
    # Process each .npz file
    for npz_file in npz_files:
        # Add suffix to filename if applicable
        if suffix:
            stem = npz_file.stem  # filename without extension
            ext = npz_file.suffix  # .npz
            new_filename = f"{stem}{suffix}{ext}"
            output_path = output_folder / new_filename
        else:
            output_path = output_folder / npz_file.name
        
        process_npz_file(npz_file, output_path)
        print()
    
    print("All files processed successfully!")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Files processed: {len(npz_files)}")

if __name__ == "__main__":
    main()
