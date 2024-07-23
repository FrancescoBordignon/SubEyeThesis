import numpy as np
import open3d as o3d
import cv2
import os
import torch
import sys
from demo import main as omniglue

# Define the remap function
def remap(value, from_range, to_range):
    (a, b) = from_range
    (c, d) = to_range
    new_value = c + (value - a) * (d - c) / (b - a)
    return new_value

# Project 3D points to the image
def project_points(points_3d, K, R, t):
    # Transform points to the camera coordinate system
    P_cam =  (R @ points_3d.T  + t[:, np.newaxis])
    
    # Project 3D points to the image plane
    P_img = K @ P_cam
    
    # Normalize by depth to get the 2D image coordinates
    P_img /= P_img[2, :]
    
    # Return only the x and y coordinates (2D)
    return P_img[:2, :].T

# INPUT PATHS
image0_path = "/home/francesco/Desktop/snapshot_known_100_depth.png"  # Replace with your reference render image path
image1_path = "/home/francesco/Desktop/depth_anything_dataset/008_depth.png"  # Replace with the folder where your sequence is
save_path = "output_opt_flow/"
ply_file = "/home/francesco/Desktop/camera_localization_dataset_00/palestrina/palestrina.ply"

torch.cuda.empty_cache()

save_name = os.path.join(save_path, "omniglue_match_*.png")

original_shape = (828, 1268)
new_shape = (400, 400)
#sys.argv = ['demo.py', image0_path, image1_path, save_name, new_shape]
#pts0, pts1 = omniglue(sys.argv)

if True:
    # Intrinsic parameters of the first camera known from meshlab
    fx1 = 900
    fy1 = fx1
    cx1 = 300
    cy1 = 200

    # Intrinsic parameters of the first camera
    K1 = np.array([[fx1, 0, cx1],
                   [0, fy1, cy1],
                   [0, 0, 1]], dtype=np.float32)

    # Define extrinsic parameters
#    R = np.array([[ 0.88501903, -0.08718199 , 0.45731798  ],
# [ 0.0240497,   0.98955899 , 0.142105            ],
# [-0.46493301 ,-0.11476699 , 0.87787598          ]], dtype=np.float32))


    R = np.array([[-1,0,0],
                [0,1,0],
                [0,0,-1]], dtype=np.float32)
    #R, inv_flag = cv2.invert(R, cv2.DECOMP_LU)
    R1 = np.array([[0.877,0,0.47],
                [0,1,0],
                [-0.47,0,0.877]], dtype=np.float32)
    
    R2 = np.array([[-1,0,0],
            [0,-1,0],
            [0,0,1]], dtype=np.float32)
    R2_2 = np.array([[0.99,-0.1,0],
            [0.1,0.99,0],
            [0,0,1]], dtype=np.float32)

    R3 = np.array([[1,0,0],
            [0,0.97,0.22],
            [0,-0.22,0.97]], dtype=np.float32)

    R = R @ R1 @ R2 @ R2_2 @ R3
    t = np.array([213, 143.411, 383], dtype=np.float32)

    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(R)

    print("Rotation vector:\n", rvec)
    # Load the 3D model
    mesh = o3d.io.read_triangle_mesh(ply_file)
    points_3d = np.asarray(mesh.vertices)
    # Create a coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15.0, origin=[0, 0, 0])

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15.0, origin=t)

    # Apply the rotation to the coordinate frame
    camera_frame.rotate(R, center=t)

    o3d.visualization.draw_geometries([mesh, coordinate_frame, camera_frame])
    # Given 2D keypoint on the image
    #keypoint_2d = np.array(pts0[0])
    #print(f"given {keypoint_2d[0]} {keypoint_2d[1]}")
    
    #keypoint_2d[0] = remap(keypoint_2d[0], (0, new_shape[0]), (0, original_shape[0]))
    #keypoint_2d[1] = remap(keypoint_2d[1], (0, new_shape[1]), (0, original_shape[1]))
    #print(f"that remapped is {keypoint_2d[0]} {keypoint_2d[1]}")

    # Project all 3D points to 2D image
    #points_2d = project_points(points_3d, K1, R, t)
    
    points_2d, _ = cv2.projectPoints(points_3d, rvec, t, K1, np.zeros((4, 1)))
    #points_2d, _ = cv2.projectPoints(points_3d, np.zeros((1, 3)), np.zeros((1, 3)), K1, np.zeros((4, 1)))
    print(len(points_2d))
    # Find the nearest 3D point
    #distances = np.linalg.norm(points_2d - keypoint_2d, axis=1)
    #nearest_index = np.argmin(distances)
    #nearest_point_3d = points_3d[0]

    #print("Nearest 3D point:", nearest_point_3d)


    # Create an empty image
    image = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)

    # Draw all points as white dots
    for point in points_2d:
        #print(str(int(point[0])) + str(int(point[1])) )
        cv2.circle(image, (int(point[0][0]), int(point[0][1])), 1, (255, 255, 255), -1)

    # Draw the nearest point as a red dot
    #nearest_point_2d = points_2d[0]
    #cv2.circle(image, (int(nearest_point_2d[0]), int(nearest_point_2d[1])), 3, (0, 0, 255), -1)
    #cv2.circle(image, (int(keypoint_2d[1]), int(keypoint_2d[0])), 3, (0, 0, 255), -1)
    cv2.circle(image, (0,0), 3, (0, 0, 255), -1)
    cv2.circle(image, (0, 20), 3, (0, 0, 255), -1)
    cv2.circle(image, (100, 20), 3, (0, 0, 255), -1)
    # Save the image
    #cv2.imwrite("projected_points.png", image)

    # Display the image
    #flipped_image = cv2.flip(image, 0)
    #flipped_image = cv2.flip(flipped_image, 1)
    cv2.imshow("Projected 2D Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Not enough points found")