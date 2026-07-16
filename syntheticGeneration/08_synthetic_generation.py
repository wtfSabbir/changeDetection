import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

def find_safe_edit_region(mask_path, box_size=256):
    """Finds a square region that is 100% visible in both camera angles."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape
    
    # We want to find a spot that is mostly road/sidewalk (bottom half of the image)
    # and completely inside the white "Safe Zone"
    for _ in range(50): # Try 50 random spots
        y = np.random.randint(h // 2, h - box_size) # Limit to bottom half
        x = np.random.randint(0, w - box_size)
        
        # Check if this entire square is white (255)
        patch = mask[y:y+box_size, x:x+box_size]
        if np.all(patch == 255):
            return x, y, box_size
            
    return None # No safe zone found big enough

def generate_synthetic_change(node_dir, pipe, device):
    """Injects a fake urban object into the base photo."""
    print(f"Synthesizing fake infrastructure for: {os.path.basename(node_dir)}")
    
    base_img_path = os.path.join(node_dir, "cubemap_base", "front.jpg")
    mask_path = os.path.join(node_dir, "cubemap_temporal_0", "mask_front.png")
    synthetic_out_path = os.path.join(node_dir, "synthetic_base_front.jpg")
    
    if os.path.exists(synthetic_out_path):
        return # Idempotency: Skip if we already generated a fake image here
        
    if not os.path.exists(base_img_path) or not os.path.exists(mask_path):
        return

    # 1. Find a physically safe spot to put the fake object
    safe_zone = find_safe_edit_region(mask_path)
    if safe_zone is None:
        print(f"  -> Skipped: No safe viewing zone found.")
        return
        
    x, y, size = safe_zone
    
    # 2. Create the AI Edit Mask (White = AI paints here, Black = AI ignores)
    img = cv2.imread(base_img_path)
    h, w = img.shape[:2]
    edit_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw the box where the AI is allowed to hallucinate
    edit_mask[y:y+size, x:x+size] = 255 
    
    # Convert to PIL for HuggingFace
    base_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(edit_mask)
    
    # 3. THE MAGIC: Tell the AI what to build
    prompt = "a bright orange traffic cone on the street, hyperrealistic, 8k resolution, perfect lighting"
    negative_prompt = "cartoon, fake, floating, badly drawn, messy"
    
    # Run the Diffusion Model
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_pil,
            mask_image=mask_pil,
            num_inference_steps=30, # Quality level
            guidance_scale=7.5
        ).images[0]
        
    # Save the brand new synthetic training image
    result.save(synthetic_out_path)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    # Hardware acceleration check
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading Stable Diffusion Inpainting on: {device.type.upper()}...")

    # Load the heavy Generative AI model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        variant="fp16"
    )
    pipe = pipe.to(device)

    # Find all GPS nodes
    node_folders = glob.glob(os.path.join(args.data_dir, "*", "node_*"))
    
    for idx, folder in enumerate(node_folders):
        print(f"Processing {idx+1}/{len(node_folders)}")
        generate_synthetic_change(folder, pipe, device)

    print("\nPhase 3 Complete: Synthetic Training Dataset Generated!")

if __name__ == "__main__":
    main()