## Dataset Citation

The dataset **mnist_v0** (renamed for uniformity) used in this project was taken from the following GitHub repository, which maintains a mirror of the original MNIST database:

	- https://github.com/cvdfoundation/mnist

The original MNIST database is hosted at: <http://yann.lecun.com/exdb/mnist/>

If you use this dataset, please cite:

```bibtex
@article{lecun2010mnist,
	title={MNIST handwritten digit database},
	author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
	journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
	volume={2},
	year={2010}
}
```
## Visualizing MNIST .idx3-ubyte Files

To help visualize the images in any `.idx3-ubyte` file (such as those in the MNIST dataset), a script named `visualize_idx3_ubyte.py` is provided in the `task1` folder of this repository.

**Location:** `task1/visualize_idx3_ubyte.py`

**Usage:**

Run the script from the command line, providing the path to your `.idx3-ubyte` file as the first argument:

```
python task1/visualize_idx3_ubyte.py <path_to_idx3_ubyte_file>
```

This will display 25 images per page in a 5x5 grid. You can navigate through the images using the right and left arrow keys to go forward and backward, respectively.

### Script Optimizations

- **Memory Efficiency:** Uses `np.memmap` to map the file directly on disk, preventing system crashes by loading only the necessary image slices into RAM.
- **Rendering Performance:** Initializes plot axes once and uses the `.set_data()` method to update images instead of clearing the entire figure with `plt.clf()`, eliminating flickering and reducing computational cost of page transitions.
- **Improved Usability:** Migrates from `sys.argv` to the `argparse` library for a more flexible command-line interface, supporting configurable parameters such as grid size and color maps.