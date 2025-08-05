import numpy as np
import matplotlib.pyplot as plt
import os

def save_prediction_map(tensor_map, save_path):
    if tensor_map.ndim == 3 and tensor_map.shape[0] == 3:
        tensor_map = np.mean(tensor_map, axis=0)  # Convert to [H, W]
    elif tensor_map.ndim == 3 and tensor_map.shape[2] == 3:
        tensor_map = np.mean(tensor_map, axis=2)

    tensor_map -= tensor_map.min()
    tensor_map /= (tensor_map.max() + 1e-8)

    viridis = plt.cm.get_cmap('viridis')
    colored_map = viridis(tensor_map)[:, :, :3]  

    img_uint8 = (colored_map * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, img_uint8)
    print(f"[Saved] Prediction image to {save_path}")
