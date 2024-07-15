#Bordignopn Francesco Unipd
#This program was created during my master thesis.
#The aim is to qualitatively asses if a matcher performs good to estimate optical flow.
#if this happens this means that a sub group of keypoints can be used to estimate the
#relative pose of the camera (final aim of the thesis).
#The two matcher used are: loftr and eloftr
#It loads a reference image and a sequence of images from a folder, then uses LoFTR or EfficientLoFTR 
#models to find keypoint matches between the reference image and the sequence images. 
#The program calculates the centroids of the matching keypoints, saves images with the matches, and 
#estimates new centroids based on previous movements and Gaussian noise. 
#Finally, it saves central batch images with valid keypoints and estimated centroids.



# LIBRARIES
import kornia.feature as KF
import kornia as K 
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy
from EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

# INPUT PATHS
image0_path = "/home/francesco/Desktop/target_palestrina/im65.png"  # Reference render image path
folder_path = "/home/francesco/Desktop/sequence1_palestrina/"       # Folder where your sequence is stored
save_path = "/home/francesco/Desktop/image/"                        # Path to save results
image_resize_dims = (540, 540)                                      # Size of the image for LoFTR
torch.cuda.empty_cache()

# PARAMETERS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use a GPU if available
confidence_thr = 0.3                 # Threshold for confidence
keypoint_good_confidence_thr = 2     # Good confidence threshold for keypoints
flow_thr = 50                        # Threshold for flow
batch_len = 7                        # Number of images in a batch
save_to_folder = True               # Save individual images
save_central = False                  # Save central images
model = "loftr"                     # Matcher to evaluate

# SUPPORT FUNCTIONS
# Define the transform to resize and convert to tensor required for LoFTR
transform = transforms.Compose([
    transforms.Resize(image_resize_dims),  # Resize the image
    transforms.ToTensor(),                 # Convert to tensor and range 0-1
])

# Function to find indices of matching keypoints
def trova_indici_minimi(tensor1, tensor2):
    tensor1_list = set(map(tuple, tensor1.cpu().numpy()))
    tensor2_list = set(map(tuple, tensor2.cpu().numpy()))
    
    min_idxs = []
    for t1 in tensor1_list:
        found = False
        for j, t2 in enumerate(tensor2_list):
            if t1 == t2:
                min_idxs.append(j)
                found = True
                break
        if not found:
            min_idxs.append(None)  
    return min_idxs

# Euclidean distance function for tensors
def euclidean_distance(point1, point2):
    return torch.norm(point1 - point2)

# Sum over list of 2D tensors
def sum_over_list(lista):
    counter = torch.zeros((len(lista[-1])), device=device)
    sums = torch.zeros((len(lista[-1]), 2), device=device)
    for i in range(len(lista)):
        for j in range(len(lista[i])):
            if lista[i][j][0] != 0 and lista[i][j][1] != 0:
                sums[j][0] += lista[i][j][0]
                sums[j][1] += lista[i][j][1]
                counter[j] += 1
    return sums, counter        

# SETUP LOFTR
if model == "loftr":
    loftr = KF.LoFTR(pretrained='outdoor').to(device)  # outdoor or indoor

# SETUP ELOFTR
if model == "eloftr":
    # You can choose model type in ['full', 'opt']
    model_type = 'full'  # 'full' for best quality, 'opt' for best efficiency
    # You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
    precision_eloftr = 'fp32'  # Choose precision

    if model_type == 'full':
        _default_cfg = deepcopy(full_default_cfg)
    elif model_type == 'opt':
        _default_cfg = deepcopy(opt_default_cfg)
        
    if precision_eloftr == 'mp':
        _default_cfg['mp'] = True
    elif precision_eloftr == 'fp16':
        _default_cfg['half'] = True

    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("EfficientLoFTR/weights/eloftr_outdoor.ckpt")['state_dict'])
    matcher = reparameter(matcher)  # Ensure reparameterization if needed

    if precision_eloftr == 'fp16':
        matcher = matcher.half()
    matcher = matcher.eval().to(device)

# PRINT SETUP
print("model-------" + model)
print("confidence--" + str(confidence_thr))
print("flow thr----" + str(flow_thr) + "px")
print("batch_len---" + str(batch_len))
if save_to_folder:
    print("save to folder all images")
if save_central:
    print("save to folder central image of the batch")

# MAIN PROGRAM
image0 = Image.open(image0_path).convert("L")  # Convert reference image to grayscale
image0_tensor = transform(image0).to(device)   # Convert to tensor and resize

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]  # List of images
images.sort()  # Sort the list of files by name

stored_kp0 = torch.empty((0, 2), device=device)
centroids = []
batches = len(images) // batch_len
print("\nprocessing...")

for batch_id in range(batches):  # Change the range if needed
    stored_kp1 = []
    historical_kp0 = []
    historical_kp1 = []
    

    
    
    for i in range(batch_len):
        # Clear GPU cache at the start of each image
        torch.cuda.empty_cache()
        image1_path = os.path.join(folder_path, images[i + batch_id * batch_len])
        
        if model == "loftr":
            image1 = Image.open(image1_path).convert("L")
            image1_tensor = transform(image1).to(device)

            image0_tensor_s = image0_tensor.unsqueeze(0)
            image1_tensor_s = image1_tensor.unsqueeze(0)

            input = {"image0": image0_tensor_s, "image1": image1_tensor_s}
            out = loftr(input)

            keypoints0 = out.get("keypoints0").to(device)
            keypoints1 = out.get("keypoints1").to(device)
            
            keypoints0 = keypoints0[out.get("confidence") > confidence_thr]
            keypoints1 = keypoints1[out.get("confidence") > confidence_thr]
        
        elif model == "eloftr":
            img0_raw = cv2.imread(image0_path, cv2.IMREAD_GRAYSCALE)
            img1_raw = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
            img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1] // 32 * 32, img0_raw.shape[0] // 32 * 32))
            img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1] // 32 * 32, img1_raw.shape[0] // 32 * 32))

            if precision_eloftr == 'fp16':
                img0 = torch.from_numpy(img0_raw)[None][None].half().to(device) / 255.0
                img1 = torch.from_numpy(img1_raw)[None][None].half().to(device) / 255.0
            else:
                img0 = torch.from_numpy(img0_raw)[None][None].to(device) / 255.0
                img1 = torch.from_numpy(img1_raw)[None][None].to(device) / 255.0

            input = {'image0': img0, 'image1': img1}

            with torch.no_grad():
                if precision_eloftr == 'mp':
                    with torch.autocast(device_type='cuda', enabled=True):
                        matcher(input)
                else:
                    matcher(input)
            out = input
            keypoints0 = out.get('mkpts0_f').to(device)
            keypoints1 = out.get('mkpts1_f').to(device)
            
            keypoints0 = keypoints0[out.get('mconf') > confidence_thr]
            keypoints1 = keypoints1[out.get('mconf') > confidence_thr]

        if i == 0:
            stored_kp0 = keypoints0
            stored_kp1.append(keypoints1.to(device))
        else:
            stored_kp1.append(torch.zeros(((stored_kp0.shape)[0], 2), device=device))
        
        idxs = trova_indici_minimi(keypoints0, stored_kp0)
        
        for k in range(len(idxs)):
            if idxs[k] is None: 
                stored_kp0 = torch.cat((stored_kp0, keypoints0[k].unsqueeze(0)), dim=0)
                stored_kp1[i] = torch.cat((stored_kp1[i], keypoints1[k].unsqueeze(0)), dim=0)
            else:
                stored_kp1[i][idxs[k]] = keypoints1[k]
        
        historical_kp1.append(keypoints1)
        historical_kp0.append(keypoints0)

    sums, counter = sum_over_list(stored_kp1)
    centroids.append(sums / counter.view(-1, 1))  # Centroids contains all the estimated centroids for each keypoint in the render and is expressed in real image coordinates (in pixels). All of this for each batch
    
    # Estimate a centroid if the number of appearances of that keypoint is too weak, i.e., < keypoint_good_confidence_thr
    if batch_id > 5:  # Wait to have enough data
        for i in range(len(counter)):
            if counter[i] < keypoint_good_confidence_thr: 
                if len(centroids[batch_id-1]) > i and len(centroids[batch_id-2]) > i:  # Verify that keypoint has been already encountered
                    centroids[batch_id][i] = centroids[batch_id-1][i] + np.clip(np.random.normal(0.5, 0.1), 0, 1) * (centroids[batch_id-1][i] - centroids[batch_id-2][i])  # Guess a new centroid based on the movement of the previous two and Gaussian noise
    
    # Save all images with matches that respond to the optical flow conditions
    if save_to_folder:  
        for i in range(batch_len):
            good_kp0 = torch.tensor((0, 2), device=device, dtype=torch.float32)
            good_kp0 = good_kp0.unsqueeze(0)
            good_kp1 = torch.tensor((0, 2), device=device, dtype=torch.float32)
            good_kp1 = good_kp1.unsqueeze(0)
            idxs = trova_indici_minimi(historical_kp0[i], stored_kp0)

            for n in range(len(historical_kp1[i])):
                if euclidean_distance(historical_kp1[i][n], centroids[batch_id][idxs[n]]) < flow_thr:
                    good_kp0 = torch.cat((good_kp0, historical_kp0[i][n].unsqueeze(0)), dim=0)
                    good_kp1 = torch.cat((good_kp1, historical_kp1[i][n].unsqueeze(0)), dim=0)
            
            image1_path = os.path.join(folder_path, images[i + batch_id * batch_len])        
            
            keypoints0 = good_kp0.cpu().detach().numpy()
            keypoints1 = good_kp1.cpu().detach().numpy()
            if model == "loftr":
                image1 = Image.open(image1_path).convert("L")
                image1_tensor = transform(image1).to(device)

                image0_tensor_s = image0_tensor.unsqueeze(0).to(device)
                image1_tensor_s = image1_tensor.unsqueeze(0).to(device)
                

                image0_np = image0_tensor.squeeze().cpu().detach().numpy()
                image1_np = image1_tensor.squeeze().cpu().detach().numpy()

                image0_np = (image0_np * 255).astype(np.uint8)
                image1_np = (image1_np * 255).astype(np.uint8)

                image0_np = np.repeat(image0_np[:, :, np.newaxis], 3, axis=2)
                image1_np = np.repeat(image1_np[:, :, np.newaxis], 3, axis=2)
            
            if model == "eloftr":
                img0_raw = cv2.imread(image0_path, cv2.IMREAD_GRAYSCALE)
                img1_raw = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
                image0_np = cv2.resize(img0_raw, (img0_raw.shape[1] // 32 * 32, img0_raw.shape[0] // 32 * 32))
                image1_np = cv2.resize(img1_raw, (img1_raw.shape[1] // 32 * 32, img1_raw.shape[0] // 32 * 32))

            matches = [cv2.DMatch(i, i, 0) for i in range(len(keypoints0))]
            output_image = cv2.drawMatches(
                image0_np,
                [cv2.KeyPoint(kp[0], kp[1], 1) for kp in keypoints0],
                image1_np,
                [cv2.KeyPoint(kp[0], kp[1], 1) for kp in keypoints1],
                matches,
                None
            )
            cv2.imwrite(os.path.join(save_path, f"im_{i + batch_id * batch_len}.png"), output_image)
    
    # Save the central image of the batch matched with the render with all the good centroids found in the batch
    if save_central:
        good_kp0 = stored_kp0[counter > keypoint_good_confidence_thr].to(device)
        good_kp1 = centroids[batch_id][counter > keypoint_good_confidence_thr].to(device)
        image1_path = os.path.join(folder_path, images[batch_id * batch_len + int(batch_len / 2)])
        
        keypoints0 = good_kp0.cpu().detach().numpy()
        keypoints1 = good_kp1.cpu().detach().numpy()
        
        if model == "loftr":
            image1 = Image.open(image1_path).convert("L")
            image1_tensor = transform(image1).to(device)

            image0_tensor_s = image0_tensor.unsqueeze(0).to(device)
            image1_tensor_s = image1_tensor.unsqueeze(0).to(device)

            image0_np = image0_tensor.squeeze().cpu().detach().numpy()
            image1_np = image1_tensor.squeeze().cpu().detach().numpy()

            image0_np = (image0_np * 255).astype(np.uint8)
            image1_np = (image1_np * 255).astype(np.uint8)

            image0_np = np.repeat(image0_np[:, :, np.newaxis], 3, axis=2)
            image1_np = np.repeat(image1_np[:, :, np.newaxis], 3, axis=2)

        if model == "eloftr":
            img0_raw = cv2.imread(image0_path, cv2.IMREAD_GRAYSCALE)
            img1_raw = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
            image0_np = cv2.resize(img0_raw, (img0_raw.shape[1] // 32 * 32, img0_raw.shape[0] // 32 * 32))
            image1_np = cv2.resize(img1_raw, (img1_raw.shape[1] // 32 * 32, img1_raw.shape[0] // 32 * 32))

        matches = [cv2.DMatch(i, i, 0) for i in range(len(keypoints0))]
        output_image = cv2.drawMatches(
            image0_np,
            [cv2.KeyPoint(kp[0], kp[1], 1) for kp in keypoints0],
            image1_np,
            [cv2.KeyPoint(kp[0], kp[1], 1) for kp in keypoints1],
            matches,
            None
        )
        cv2.imwrite(os.path.join(save_path, f"im_{batch_id}.png"), output_image)

print("end of the program")
