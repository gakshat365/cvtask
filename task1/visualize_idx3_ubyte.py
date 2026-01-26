import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_idx_header(f):
    magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
    if magic != 2051:
        raise ValueError(f"Invalid magic number {magic}")
    return num_images, rows, cols

def visualize_idx3_ubyte(file_path, per_page=25):
    with open(file_path, 'rb') as f:
        num_images, rows, cols = read_idx_header(f)
        # Optimization 1: Use memmap to avoid loading everything into RAM
        images = np.memmap(file_path, dtype=np.uint8, mode='r', 
                           offset=16, shape=(num_images, rows, cols))

    side = int(np.sqrt(per_page))
    fig, axes = plt.subplots(side, side, figsize=(8, 8))
    axes = axes.flatten()
    ims = []

    # Optimization 2: Initialize image objects once
    for i in range(per_page):
        im = axes[i].imshow(np.zeros((rows, cols)), cmap='gray', vmin=0, vmax=255)
        axes[i].axis('off')
        ims.append(im)

    state = {'current': 0}

    def update_display():
        start = state['current']
        for i, im in enumerate(ims):
            idx = start + i
            if idx < num_images:
                im.set_data(images[idx])
                im.set_visible(True)
            else:
                im.set_visible(False)
        fig.suptitle(f"Images {start + 1} to {min(start + per_page, num_images)} of {num_images}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right' and state['current'] + per_page < num_images:
            state['current'] += per_page
            update_display()
        elif event.key == 'left' and state['current'] - per_page >= 0:
            state['current'] -= per_page
            update_display()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_display()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast IDX3-UBYTE Visualizer")
    parser.add_argument("file", help="Path to the idx3-ubyte file")
    parser.add_argument("--grid", type=int, default=25, help="Images per page (perfect square recommended)")
    args = parser.parse_args()
    
    visualize_idx3_ubyte(args.file, args.grid)