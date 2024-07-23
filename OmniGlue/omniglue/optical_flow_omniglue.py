import sys
import os
from demo import main as omniglue
import torch
import cv2
import numpy as np


#INPUT PATHS
image0_path = "/home/francesco/Desktop/snapshot_known_100_depth.png"                  # Replace with your reference render image path
folder_path = "/home/francesco/Desktop/depth_anything_dataset/"              # Replace with the folder where your sequence is
save_path = "output_opt_flow/"



images = os.listdir(folder_path)
images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
images.sort() # Sort the list of files by name



for i in range(len(images)):
    torch.cuda.empty_cache()
    image1_path = folder_path + images[i]
    save_name = save_path + "omniglue_match_" + str(i) + ".png"
    sys.argv = ['demo.py', image0_path, image1_path, save_name]
    pts0, pts1 = omniglue(sys.argv)

    if(len(pts0)>15):

        # Intrinsic parameters of the first camera known from meshlab
        fx1 = 2011.43
        fy1 = 2011.43
        cx1 = 414
        cy1 = 634
        # Intrinsic parameters of the first camera
        K1 = np.array([[fx1, 0, cx1],
                    [0, fy1, cy1],
                    [0, 0, 1]])

        # Compute the Fundamental Matrix
        F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC)

        # Compute the Essential Matrix
        E = K1.T @ F @ K1

        # Decompose the Essential Matrix into R and t
        _, R, t, mask = cv2.recoverPose(E, pts0, pts1, K1)

        # R and t now contain the rotation and translation from the first camera to the second camera

        print("Rotation matrix: \n", R)
        print("Translation vector: \n", t)

        # Now you can use R and t as the extrinsic parameters of the second camera relative to the first camera