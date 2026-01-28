import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

class DatasetVisualizer:
    def __init__(self, npz_file):
        """Initialize the visualizer with NPZ file."""
        print(f"Loading {npz_file}...")
        data = np.load(npz_file)
        self.images = data['images']
        self.labels = data['labels']
        self.num_images = len(self.images)
        
        # Display settings
        self.images_per_page = 20
        self.grid_rows = 4
        self.grid_cols = 5  # 4x5 grid
        self.current_index = 0
        
        # Setup figure
        self.fig, self.axes = plt.subplots(
            self.grid_rows, self.grid_cols, 
            figsize=(12, 10)
        )
        self.fig.canvas.manager.set_window_title(f'Dataset Visualizer - {npz_file}')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        print(f"Loaded {self.num_images} images")
        print(f"Image shape: {self.images.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print("\nControls:")
        print("  Right Arrow / Space: Next page")
        print("  Left Arrow: Previous page")
        print("  Q / Escape: Quit")
        
    def display_page(self):
        """Display current page of 20 images."""
        start_idx = self.current_index
        end_idx = min(start_idx + self.images_per_page, self.num_images)
        
        # Clear all axes
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                self.axes[i, j].clear()
                self.axes[i, j].axis('off')
        
        # Display images
        for idx in range(self.images_per_page):
            img_idx = start_idx + idx
            
            if img_idx >= self.num_images:
                break
            
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            
            # Get image and label
            img = self.images[img_idx]
            label = self.labels[img_idx]
            
            # Display image (convert 3-channel grayscale to regular grayscale for display)
            # Since all channels are the same, just take the first channel
            img_gray = img[:, :, 0]
            
            self.axes[row, col].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
            self.axes[row, col].set_title(f'Label: {label}', fontsize=10, pad=5)
            self.axes[row, col].axis('off')
        
        # Update title with page info
        page_num = (start_idx // self.images_per_page) + 1
        total_pages = (self.num_images + self.images_per_page - 1) // self.images_per_page
        self.fig.suptitle(
            f'Images {start_idx + 1}-{end_idx} of {self.num_images} | Page {page_num}/{total_pages}',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def on_key_press(self, event: KeyEvent):
        """Handle keyboard events for navigation."""
        if event.key in ['right', ' ']:
            # Next page
            if self.current_index + self.images_per_page < self.num_images:
                self.current_index += self.images_per_page
                self.display_page()
        
        elif event.key == 'left':
            # Previous page
            if self.current_index - self.images_per_page >= 0:
                self.current_index -= self.images_per_page
                self.display_page()
        
        elif event.key in ['q', 'escape']:
            # Quit
            plt.close(self.fig)
    
    def show(self):
        """Display the visualization."""
        self.display_page()
        plt.show()


if __name__ == "__main__":
    import sys
    
    # Default to training data if no argument provided
    if len(sys.argv) > 1:
        npz_file = sys.argv[1]
    else:
        npz_file = 'data/train_data.npz'
        print(f"No file specified, using default: {npz_file}")
        print(f"Usage: python visualize_npz.py <npz_file>\n")
    
    # Create and show visualizer
    visualizer = DatasetVisualizer(npz_file)
    visualizer.show()
