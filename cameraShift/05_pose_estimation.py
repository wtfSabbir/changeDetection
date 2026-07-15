import os
import glob
import json
import cv2
import numpy as np
from multiprocessing import Pool

def calculate_relative_pose(img1_path, img2_path):
    """Calculates the 3D translation and rotation between two images."""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return None

    # 1. Feature Extraction (Finding unique points like window corners)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None

    # 2. Feature Matching (Finding where those points went in the new photo)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's Ratio Test to keep only the absolute best matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        return None # Not enough points to calculate 3D geometry

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # 3. The Math: Camera Intrinsics
    # Since we unfolded a 360 sphere into 4 faces, each face is exactly 90 degrees FOV.
    # For a 1024x1024 image with a 90 deg FOV, the focal length is exactly half the width.
    focal_length = 512.0
    center = (512.0, 512.0)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype=np.float64)

    # 4. Calculate Essential Matrix & Recover 3D Pose
    E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return None

    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix)

    return {
        "rotation": R.tolist(),
        "translation": t.tolist()
    }

def process_node(node_dir):
    """Aligns the temporal photos to the base photo for a single GPS pin."""
    pose_file = os.path.join(node_dir, "camera_poses.json")
    if os.path.exists(pose_file):
        return # Idempotent: Skip if we already calculated this node
        
    print(f"Calculating 3D Pose for: {os.path.basename(node_dir)}")
    
    base_front = os.path.join(node_dir, "cubemap_base", "front.jpg")
    if not os.path.exists(base_front):
        return

    node_poses = {}
    
    # Compare every historical drive to the modern base drive
    temporal_dirs = glob.glob(os.path.join(node_dir, "cubemap_temporal_*"))
    for t_dir in temporal_dirs:
        t_front = os.path.join(t_dir, "front.jpg")
        t_name = os.path.basename(t_dir).replace("cubemap_", "")
        
        pose_data = calculate_relative_pose(t_front, base_front)
        if pose_data:
            node_poses[t_name] = pose_data
            
    # Save the physics data
    if node_poses:
        with open(pose_file, 'w') as f:
            json.dump(node_poses, f, indent=4)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    node_folders = glob.glob(os.path.join(args.data_dir, "*", "node_*"))
    print(f"Found {len(node_folders)} nodes. Initiating Epipolar Geometry calculations...")
    
    with Pool(processes=8) as pool:
        pool.map(process_node, node_folders)
        
    print("\nPhase 2.2 Complete: Camera translation vectors mapped!")

if __name__ == "__main__":
    main()