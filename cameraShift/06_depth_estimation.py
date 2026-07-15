import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation

def generate_depth_map(image_path, processor, model, device):
    """Generates a depth map from a 2D image and saves it as raw data and a visualization."""
    if not os.path.exists(image_path):
        return

    # Set up file paths
    dir_name = os.path.dirname(image_path)
    base_name = os.path.basename(image_path).split('.')[0] # e.g., 'front'
    
    npy_path = os.path.join(dir_name, f"depth_{base_name}.npy")
    png_path = os.path.join(dir_name, f"depth_{base_name}.png")
    
    # Idempotency: Skip if already processed
    if os.path.exists(npy_path):
        return

    # 1. Load Image
    image = Image.open(image_path).convert("RGB")

    # 2. Prepare and run through the AI model
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # 3. Interpolate back to original image resolution (1024x1024)
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    
    # Extract raw numpy array
    depth_array = prediction.squeeze().cpu().numpy()

    # 4. Save the raw math array for 3D reprojection (The Brains)
    np.save(npy_path, depth_array)

    # 5. Save a pretty visualization for humans (The Looks)
    depth_min = depth_array.min()
    depth_max = depth_array.max()
    depth_normalized = (255 * (depth_array - depth_min) / (depth_max - depth_min)).astype(np.uint8)
    
    # Apply a colormap (INFERNO) so close objects look bright yellow/white, far objects look dark purple
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    cv2.imwrite(png_path, depth_colored)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    # Hardware acceleration check (CUDA for NVIDIA, MPS for Mac, CPU as fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading AI Depth Model on: {device.type.upper()}...")

    # Load the MiDaS/DPT Model from HuggingFace
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
    model.eval()

    # Find all unfolded cubemap directories
    cubemap_folders = glob.glob(os.path.join(args.data_dir, "*", "node_*", "cubemap_*"))
    print(f"Found {len(cubemap_folders)} camera positions. Generating 3D Depth Maps...")

    # We iterate sequentially here because running multiple massive AI models in parallel usually crashes GPU RAM
    for idx, folder in enumerate(cubemap_folders):
        print(f"Processing {idx+1}/{len(cubemap_folders)}: {os.path.basename(os.path.dirname(folder))}/{os.path.basename(folder)}")
        
        # We primarily only need depth for the Front face, as that is where our forward optical flow is
        front_img = os.path.join(folder, "front.jpg")
        generate_depth_map(front_img, processor, model, device)

    print("\nPhase 2.3 Complete: 3D Depth Maps Generated!")

if __name__ == "__main__":
    main()