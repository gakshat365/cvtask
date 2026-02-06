import numpy as np
import struct

def read_idx_images(filename):
    with open(filename, 'rb') as f:
        f.read(4)  # skip magic
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    return images

def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)  # skip magic and count
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def convert_dataset(img_file, label_file, output_file):
    images = read_idx_images(img_file)
    labels = read_idx_labels(label_file)
    
    # Convert to 3-channel in one operation
    images_3ch = np.repeat(images[:, :, :, np.newaxis], 3, axis=3)
    
    np.savez(output_file, images=images_3ch, labels=labels)

if __name__ == "__main__":
    convert_dataset('data/train_img.idx3-ubyte', 'data/train_label.idx1-ubyte', 'data/train_data.npz')
    convert_dataset('data/test_img.idx3-ubyte', 'data/test_label.idx1-ubyte', 'data/test_data.npz')
