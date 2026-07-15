import os
import glob
import json
import cv2
import numpy as np
from multiprocessing import Pool

def warp_and_mask(img_t_path, depth_t_path, pose, img_base_path, output_mask_path):
    """Warps temporal image to base perspective and generates a disocclusion mask."""
    if not os.path.exists(img_t_path) or not os.path.exists(depth_t_path):
        return

    # Load 2D Temporal Image and 3D Depth Array
    img_t = cv2.imread(img_t_path)
    depth_t = np.load(depth_t_path)
    
    h, w = img_t.shape[:2]
    
    # Camera Intrinsics (90 deg FOV for a 1024x1024 cubemap face)
    focal_length = w / 2.0
    cx, cy = w / 2.0, h / 2.0
    
    # 1. THE UNPROJECTION: Convert 2D pixels into a 3D Point Cloud
    # Create a grid of every pixel coordinate (u, v)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Math: X = (u - cx) * Z / f, Y = (v - cy) * Z / f
    Z = depth_t
    X = (u - cx) * Z / focal_length
    Y = (v - cy) * Z / focal_length
    
    # Flatten into a list of 3D points: Shape (3, N)
    points_3d = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))
    
    # 2. THE TRANSFORMATION: Move the points using MapAnything Pose Vectors
    # Parse rotation and translation from the JSON file
    R = np.array(pose['rotation'])
    t = np.array(pose['translation'])
    
    # Math: P_new = R * P_old + t
    points_3d_new = R @ points_3d + t
    
    X_new, Y_new, Z_new = points_3d_new
    
    # 3. THE REPROJECTION: Flatten the points back into the Base Camera's 2D view
    # Avoid dividing by zero for points that somehow ended up behind the camera
    Z_new[Z_new <= 0] = 1e-6 
    
    u_new = (X_new * focal_length / Z_new) + cx
    v_new = (Y_new * focal_length / Z_new) + cy
    
    u_new = np.round(u_new).astype(int)
    v_new = np.round(v_new).astype(int)
    
    # 4. CREATE THE DISOCCLUSION MASK
    # Create a blank black canvas
    mask = np.zeros((h, w), dtype=np.uint8)
    warped_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Find all points that successfully landed inside the boundaries of the new screen
    valid_pts = (u_new >= 0) & (u_new < w) & (v_new >= 0) & (v_new < h)
    
    u_valid = u_new[valid_pts]
    v_valid = v_new[valid_pts]
    
    # Paint the successful pixels white on the mask, and transfer the color
    mask[v_valid, u_valid] = 255
    
    # Basic Z-buffering/Splatting (approximate)
    flat_img = img_t.reshape(-1, 3)
    warped_img[v_valid, u_valid] = flat_img[valid_pts]
    
    # Morphological closing to fill tiny 1-pixel rounding holes
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Save the files
    cv2.imwrite(output_mask_path, mask)
    cv2.imwrite(output_mask_path.replace("mask_", "warped_"), warped_img)

def process_node(node_dir):
    """Calculates spatial masks for a specific GPS pin."""
    pose_file = os.path.join(node_dir, "camera_poses.json")
    if not os.path.exists(pose_file):
        return
        
    with open(pose_file, 'r') as f:
        poses = json.load(f)
        
    print(f"Projecting Point Clouds for: {os.path.basename(node_dir)}")
    
    base_front = os.path.join(node_dir, "cubemap_base", "front.jpg")
    
    for t_name, pose in poses.items():
        t_front = os.path.join(node_dir, f"cubemap_{t_name}", "front.jpg")
        t_depth = os.path.join(node_dir, f"cubemap_{t_name}", "depth_front.npy")
        output_mask = os.path.join(node_dir, f"cubemap_{t_name}", "mask_front.png")
        
        if os.path.exists(output_mask):
            continue # Idempotent skip
            
        warp_and_mask(t_front, t_depth, pose, base_front, output_mask)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    node_folders = glob.glob(os.path.join(args.data_dir, "*", "node_*"))
    print(f"Initiating 3D Disocclusion Masking...")
    
    with Pool(processes=8) as pool:
        pool.map(process_node, node_folders)
        
    print("\nPhase 2 Complete: Geometry pipeline successfully finished!")

if __name__ == "__main__":
    main()