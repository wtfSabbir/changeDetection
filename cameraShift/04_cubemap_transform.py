import os
import glob
import cv2
import py360convert
import argparse
from multiprocessing import Pool

def unfold_panorama(image_path, output_dir):
    """Mathematically projects a 360 sphere onto flat perspective planes."""
    if not os.path.exists(image_path):
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Idempotency: skip if we've already unfolded this specific image
    if os.path.exists(os.path.join(output_dir, "front.jpg")):
        return

    # 1. Load the heavy equirectangular image
    img = cv2.imread(image_path)
    if img is None:
        return
        
    # 2. Set perspective face size (1024x1024 is ideal for MapAnything)
    face_w = 1024 
    
    # 3. The Math: Project the sphere into a dictionary of 6 cube faces
    # 'F' (Front), 'R' (Right), 'B' (Back), 'L' (Left), 'U' (Up/Sky), 'D' (Down/Road)
    cubemap_dict = py360convert.e2c(img, face_w=face_w, cube_format='dict')

    # For urban change detection, the sky and the asphalt directly under the car 
    # are useless for pose estimation. We only extract the horizontal ring.
    faces_to_save = {
        'front.jpg': cubemap_dict['F'],
        'right.jpg': cubemap_dict['R'],
        'back.jpg': cubemap_dict['B'],
        'left.jpg': cubemap_dict['L']
    }

    # Save the flattened perspective images
    for filename, face_img in faces_to_save.items():
        cv2.imwrite(os.path.join(output_dir, filename), face_img)

def process_node(node_dir):
    """Finds all panoramas inside a specific GPS node and unfolds them."""
    print(f"Unfolding geometry for: {os.path.basename(node_dir)}")
    
    # Unfold the 'After' photo
    base_path = os.path.join(node_dir, "base.jpg")
    unfold_panorama(base_path, os.path.join(node_dir, "cubemap_base"))

    # Unfold all the 'Before' photos
    temporal_paths = glob.glob(os.path.join(node_dir, "temporal_*.jpg"))
    for t_path in temporal_paths:
        t_name = os.path.basename(t_path).split('.')[0] # e.g., 'temporal_0'
        unfold_panorama(t_path, os.path.join(node_dir, f"cubemap_{t_name}"))

def main():
    parser = argparse.ArgumentParser(description="Unfold 360 Panoramas into Cubemaps.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to your city folders")
    args = parser.parse_args()

    # Find every single node folder across all cities
    node_folders = glob.glob(os.path.join(args.data_dir, "*", "node_*"))
    
    print(f"Found {len(node_folders)} total nodes. Initiating geometric unfolding...")
    
    # Use parallel processing because Matrix multiplication on thousands of HD images is heavy!
    with Pool(processes=8) as pool:
        pool.map(process_node, node_folders)
        
    print("\nPhase 2.1 Complete: All spheres successfully unfolded into perspective planes.")

if __name__ == "__main__":
    main()