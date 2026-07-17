import os
import glob
import json
import random
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# The randomized urban changes we want to inject
PROMPT_POOL = [
    "a bright orange traffic cone on the street, hyperrealistic, 8k resolution, perfect lighting",
    "a realistic pothole in the asphalt, damaged road, 8k resolution, photorealistic",
    "a white painted pedestrian crosswalk marking on the road, hyperrealistic",
    "a red and white plastic construction barrier on the road, photorealistic",
    "a dark green metal trash can sitting on the ground, hyperrealistic",
    "a yellow wet floor sign sitting on the ground, highly detailed, photorealistic",
    "a large cardboard box sitting on the street, photorealistic, 8k"
]

def find_safe_edit_region(mask_path, box_size=256):
    """Finds a safe square region strictly in the BOTTOM-CENTER (the road)."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
        
    h, w = mask.shape
    
    # Lock the search exclusively to the center 1/3rd of the screen 
    # to guarantee we are targeting the road, not the walls or fields.
    x_min = w // 3               # Left boundary of the center lane
    x_max = (2 * w) // 3 - box_size  # Right boundary
    
    y_min = h // 2               # Horizon line
    y_max = h - box_size         # Bottom of the image
    
    # If the safe zone boundaries are completely invalid, skip
    if x_min >= x_max or y_min >= y_max:
        return None

    # Search for a 100% visible spot strictly within the road boundaries
    for _ in range(100): 
        y = np.random.randint(y_min, y_max) 
        x = np.random.randint(x_min, x_max)
        
        patch = mask[y:y+box_size, x:x+box_size]
        if np.all(patch == 255):
            return x, y, box_size
            
    # If the Phase 2 overlap mask is too small in the road area, it safely skips 
    # the image instead of generating garbage on a wall.
    return None

def generate_synthetic_change(node_dir, pipe, device):
    """Injects a random fake urban object and saves the metadata."""
    print(f"Synthesizing fake infrastructure for: {os.path.basename(node_dir)}")
    
    base_img_path = os.path.join(node_dir, "cubemap_base", "front.jpg")
    mask_path = os.path.join(node_dir, "cubemap_temporal_0", "mask_front.png")
    synthetic_out_path = os.path.join(node_dir, "synthetic_base_final.jpg")
    metadata_out_path = os.path.join(node_dir, "synthetic_metadata.json")
    
    if os.path.exists(synthetic_out_path) and os.path.exists(metadata_out_path):
        return 
        
    if not os.path.exists(base_img_path) or not os.path.exists(mask_path):
        return

    safe_zone = find_safe_edit_region(mask_path)
    if safe_zone is None:
        print(f"  -> Skipped: No safe viewing zone found.")
        return
        
    x, y, size = safe_zone
    
    img = cv2.imread(base_img_path)
    h, w = img.shape[:2]
    edit_mask = np.zeros((h, w), dtype=np.uint8)
    edit_mask[y:y+size, x:x+size] = 255 
    
    base_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(edit_mask)
    
    # Pick a random urban object from the pool
    selected_prompt = random.choice(PROMPT_POOL)
    negative_prompt = "cartoon, fake, floating, badly drawn, messy, unrealistic perspective"
    
    with torch.no_grad():
        result = pipe(
            prompt=selected_prompt,
            negative_prompt=negative_prompt,
            image=base_pil,
            mask_image=mask_pil,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
    result.save(synthetic_out_path)
    
    # Save the exact coordinates and prompt so the CYWS-3D training loop can use it
    metadata = {
        "prompt": selected_prompt,
        "bounding_box": {"x": x, "y": y, "width": size, "height": size}
    }
    with open(metadata_out_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading Stable Diffusion Inpainting on: {device.type.upper()}...")

    # Final, secure model loading parameters
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        variant="fp16"
    )
    pipe = pipe.to(device)

    node_folders = glob.glob(os.path.join(args.data_dir, "*", "node_*"))
    
    for idx, folder in enumerate(node_folders):
        print(f"Processing {idx+1}/{len(node_folders)}")
        generate_synthetic_change(folder, pipe, device)

    print("\nPhase 3 Complete: Final Synthetic Training Dataset Generated!")

if __name__ == "__main__":
    main()