import numpy as np
import matplotlib.pyplot as plt
import sys

class Viewer:
    def __init__(self, file):
        data = np.load(file)
        self.imgs = data['images']
        self.labels = data['labels']
        self.total = len(self.imgs)
        self.idx = 0
        self.per_page = 20
        
        self.fig, self.axes = plt.subplots(4, 5, figsize=(12, 10))
        self.fig.canvas.manager.set_window_title(f'Viewer - {file}')
        self.fig.canvas.mpl_connect('key_press_event', self.key)
        
    def show_page(self):
        start = self.idx
        end = min(start + self.per_page, self.total)
        
        # Clear grid
        for ax in self.axes.flat:
            ax.clear()
            ax.axis('off')
        
        # Show images
        for i in range(self.per_page):
            pos = start + i
            if pos >= self.total:
                break
            
            ax = self.axes.flat[i]
            ax.imshow(self.imgs[pos])
            ax.set_title(f'{self.labels[pos]}', fontsize=10)
            ax.axis('off')
        
        # Update title
        page = (start // self.per_page) + 1
        pages = (self.total + self.per_page - 1) // self.per_page
        self.fig.suptitle(f'{start + 1}-{end} of {self.total} | Page {page}/{pages}', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def key(self, event):
        if event.key in ['right', ' ']:
            if self.idx + self.per_page < self.total:
                self.idx += self.per_page
                self.show_page()
        
        elif event.key == 'left':
            if self.idx >= self.per_page:
                self.idx -= self.per_page
                self.show_page()
        
        elif event.key in ['q', 'escape']:
            plt.close(self.fig)
    
    def show(self):
        self.show_page()
        plt.show()

if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else 'data/train_data.npz'
    Viewer(file).show()
