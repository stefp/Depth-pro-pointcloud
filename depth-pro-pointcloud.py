# Import necessary libraries
import numpy as np
from PIL import Image
import torch
import cv2
import laspy
import csv
import os
import argparse

# Import custom depth estimation module
import depth_pro  # Ensure this module provides the required functions

def tensor_to_image(tensor, output_path):
    """
    Converts a 2D PyTorch tensor to an image and saves it with float precision.

    Args:
        tensor (torch.Tensor): 2D tensor representing the image.
        output_path (str): File path to save the image.
    """
    # Ensure the tensor is on the CPU and convert to NumPy array
    tensor = tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor
    # Save the image with floating-point precision
    img = Image.fromarray(tensor.astype(np.float32))
    img.save(output_path, format='TIFF')
    print(f"Depth image saved to {output_path}")

def depth_to_point_cloud_with_rgb(depth_image, rgb_image, fx, fy, cx, cy, max_range=50):
    """
    Converts a depth image to a 3D point cloud with RGB information.

    Args:
        depth_image (numpy.ndarray): 2D array of depth values.
        rgb_image (numpy.ndarray): 3D array of RGB values.
        fx (float): Focal length in x-direction (pixels).
        fy (float): Focal length in y-direction (pixels).
        cx (float): Principal point x-coordinate.
        cy (float): Principal point y-coordinate.
        max_range (float): Maximum depth range to consider.

    Returns:
        numpy.ndarray: N x 6 array of point cloud data with XYZRGB values.
    """
    height, width = depth_image.shape
    points = []

    # Iterate over each pixel to compute 3D coordinates and retrieve RGB values
    for v in range(height):
        for u in range(width):
            Z = depth_image[v, u]  # Depth at pixel (u, v)
            if Z == 0 or Z > max_range:
                continue  # Skip invalid depth values
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            # Swap Y and Z, then flip X and Z for 180-degree rotation around Y-axis
            point = [-X, Z, -Y]
            # Get RGB values
            R, G, B = rgb_image[v, u]
            points.append(point + [R, G, B])

    return np.array(points)

def save_point_cloud_as_csv(points_xyz, file_name="point_cloud.csv"):
    """
    Saves a point cloud with XYZ coordinates to a CSV file.

    Args:
        points_xyz (numpy.ndarray): N x 3 array of point cloud data with XYZ values.
        file_name (str): Output file name.
    """
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['X', 'Y', 'Z'])  # Write header
        writer.writerows(points_xyz)
    print(f"Point cloud XYZ data saved to {file_name}")

def save_point_cloud_as_laz_with_rgb(points_rgb, file_name="point_cloud.laz"):
    """
    Saves a point cloud with RGB values to a LAZ file using laspy.

    Args:
        points_rgb (numpy.ndarray): N x 6 array of point cloud data with XYZRGB.
        file_name (str): Output file name.
    """
    # Create LAS header with point format that supports RGB (2 or 3)
    header = laspy.LasHeader(point_format=3, version="1.2")
    # Set the scale factors and offsets
    header.offsets = np.min(points_rgb[:, :3], axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])  # Adjust scale for precision

    # Create LasData object and assign points
    las = laspy.LasData(header)
    # Assign X, Y, Z directly as they are already swapped and rotated in the point generation
    las.x = points_rgb[:, 0]
    las.y = points_rgb[:, 1]
    las.z = points_rgb[:, 2]
    # Scale RGB values from 0-255 to 0-65535
    las.red = (points_rgb[:, 3].astype(np.uint16) * 256).clip(0, 65535)
    las.green = (points_rgb[:, 4].astype(np.uint16) * 256).clip(0, 65535)
    las.blue = (points_rgb[:, 5].astype(np.uint16) * 256).clip(0, 65535)

    # Write the point cloud to a LAZ file
    with laspy.open(file_name, mode="w", header=header) as writer:
        writer.write_points(las.points)
    print(f"Point cloud saved to {file_name}")

def main():
    parser = argparse.ArgumentParser(description="Depth estimation and point cloud generation script.")
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    args = parser.parse_args()

    image_path = args.image_path

    # Check if the image exists
    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    # Derive output paths
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    dir_name = os.path.dirname(image_path)
    depth_output_path = os.path.join(dir_name, f"{base_name}_depth.tiff")
    point_cloud_csv_output_path = os.path.join(dir_name, f"{base_name}_pc.csv")
    point_cloud_output_path = os.path.join(dir_name, f"{base_name}_pc.laz")

    # Load depth estimation model and preprocessing transforms
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    # Load and preprocess the image
    image, _, f_px = depth_pro.load_rgb(image_path)
    image_transformed = transform(image)

    # Run depth estimation inference
    prediction = model.infer(image_transformed, f_px=f_px)
    depth = prediction["depth"]  # Depth in meters
    focal_length_px = prediction["focallength_px"]

    # Save the depth image
    tensor_to_image(depth, output_path=depth_output_path)

    # Read the depth image
    depth_image = cv2.imread(depth_output_path, cv2.IMREAD_UNCHANGED)

    # Camera intrinsic parameters
    fx = float(focal_length_px)
    fy = float(focal_length_px)
    cx = depth_image.shape[1] / 2
    cy = depth_image.shape[0] / 2

    # Convert original image to numpy array
    if isinstance(image, Image.Image):
        rgb_image = np.array(image)
    elif isinstance(image, torch.Tensor):
        rgb_image = image.cpu().numpy().transpose(1, 2, 0)
    elif isinstance(image, np.ndarray):
        rgb_image = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Ensure rgb_image matches depth_image dimensions
    if rgb_image.shape[:2] != depth_image.shape[:2]:
        rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))

    # Generate point cloud with RGB information
    point_cloud_rgb = depth_to_point_cloud_with_rgb(depth_image, rgb_image, fx, fy, cx, cy)

    # Save the point cloud with RGB as a .laz file
    save_point_cloud_as_laz_with_rgb(point_cloud_rgb, file_name=point_cloud_output_path)

if __name__ == "__main__":
    main()
