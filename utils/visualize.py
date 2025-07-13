import numpy as np
import matplotlib.pyplot as plt
import os

def save_prediction_map(tensor_map, save_path):
    """
    Save a 2D tensor (e.g., predicted pathloss map) as a viridis-colored PNG image.
    Args:
        tensor_map: numpy array of shape [H, W] or [3, H, W] or [H, W, 3]
        save_path: path to save the image (should end with .png)
    """
    # Handle shape and normalize
    if tensor_map.ndim == 3 and tensor_map.shape[0] == 3:
        tensor_map = np.mean(tensor_map, axis=0)  # Convert to [H, W]
    elif tensor_map.ndim == 3 and tensor_map.shape[2] == 3:
        tensor_map = np.mean(tensor_map, axis=2)

    # Normalize to [0, 1] for visualization
    tensor_map -= tensor_map.min()
    tensor_map /= (tensor_map.max() + 1e-8)

    # Apply viridis colormap
    viridis = plt.cm.get_cmap('viridis')
    colored_map = viridis(tensor_map)[:, :, :3]  # Drop alpha channel

    # Convert to uint8
    img_uint8 = (colored_map * 255).astype(np.uint8)

    # Save image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, img_uint8)
    print(f"[Saved] Prediction image to {save_path}")
