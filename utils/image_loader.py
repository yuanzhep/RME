import torch
from PIL import Image
import torchvision.transforms as T

transform_input = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),  
])

transform_output = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(), 
])

def load_input_output_image_pair(input_path, output_path):
    input_img = Image.open(input_path).convert("RGB")
    input_tensor = transform_input(input_img)  # shape: [3, 256, 256]

    output_img = Image.open(output_path).convert("L")  # shape: (H, W)
    output_tensor = transform_output(output_img)  # shape: [1, H, W]
    output_tensor = output_tensor.repeat(3, 1, 1)  

    input_tensor = input_tensor.unsqueeze(0)   # [1, 3, 256, 256]
    output_tensor = output_tensor.unsqueeze(0) # [1, 3, 256, 256]
    return input_tensor, output_tensor
