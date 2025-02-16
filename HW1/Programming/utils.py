import math
from matplotlib import pyplot as plt
import struct
import os
import numpy as np

def plot_cluster_centers(centers, title, k):
    n_cols = math.ceil(math.sqrt(k))
    n_rows = math.ceil(k / n_cols)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    if k == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    
    # Plot each cluster center
    for i in range(k):
        axs[i].imshow(centers[i].reshape(28, 28), cmap='gray')
        axs[i].set_title(f"Center {i}")
        axs[i].axis('off')
    
    for i in range(k, len(axs)):
        axs[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def load_images_labels(directory):
    """
    Load the handwritten images and labels from the given directory.
    The directory is expected to contain:
      - 'train-images.idx3-ubyte'
      - 'train-labels.idx1-ubyte'
    Returns:
      images: (N, 784) array of pixel data, each row is one image
      labels: (N,) array of integer labels [0..9]
    """
    # Load training images
    with open(os.path.join(directory, 'train-images-filtered.idx3-ubyte'), 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        rows, cols = struct.unpack('>II', f.read(8))
        # Read the image data as a 1D array of bytes, then reshape
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows*cols)
    
    # Load training labels
    with open(os.path.join(directory, 'train-labels-filtered.idx1-ubyte'), 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    
    return images, labels
