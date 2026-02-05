import numpy as np
import struct

def read_idx_images(filename):
    """Read IDX3-ubyte image file format."""
    with open(filename, 'rb') as f:
        # Read magic number and dimensions
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)
        
    return images

def read_idx_labels(filename):
    """Read IDX1-ubyte label file format."""
    with open(filename, 'rb') as f:
        # Read magic number and number of labels
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels

def grayscale_to_3channel(images):
    """Convert grayscale images to 3-channel grayscale (RGB where R=G=B)."""
    # Stack the same grayscale values across 3 channels
    images_3channel = np.stack([images, images, images], axis=-1)
    return images_3channel

def convert_dataset(img_file, label_file, output_file):
    """Convert IDX dataset to NPZ format with 3-channel grayscale."""
    print(f"Reading {img_file}...")
    images = read_idx_images(img_file)
    
    print(f"Reading {label_file}...")
    labels = read_idx_labels(label_file)
    
    print(f"Converting to 3-channel grayscale...")
    images_3channel = grayscale_to_3channel(images)
    
    print(f"Saving to {output_file}...")
    np.savez(output_file, images=images_3channel, labels=labels)
    
    print(f"Done! Shape: {images_3channel.shape}, Labels: {labels.shape}")
    print(f"Sample pixel value check - Original: {images[0, 0, 0]}, 3-channel: {images_3channel[0, 0, 0]}")

if __name__ == "__main__":
    # Convert training dataset
    convert_dataset('data/train_img.idx3-ubyte', 'data/train_label.idx1-ubyte', 'data/train_data.npz')
    
    # Convert test dataset
    convert_dataset('data/test_img.idx3-ubyte', 'data/test_label.idx1-ubyte', 'data/test_data.npz')
    
    # Verify the conversion
    print("\n--- Verification ---")
    train_data = np.load('data/train_data.npz')
    test_data = np.load('data/test_data.npz')
    
    print(f"Training images shape: {train_data['images'].shape}")
    print(f"Training labels shape: {train_data['labels'].shape}")
    print(f"Test images shape: {test_data['images'].shape}")
    print(f"Test labels shape: {test_data['labels'].shape}")
    
    # Verify a few pixel conversions
    print("\nSample pixel conversions:")
    sample_img = train_data['images'][0]
    for i in range(min(3, sample_img.shape[0])):
        for j in range(min(3, sample_img.shape[1])):
            pixel = sample_img[i, j]
            print(f"  Position ({i},{j}): RGB = {pixel}")
