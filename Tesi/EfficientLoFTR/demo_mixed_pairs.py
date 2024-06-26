import os
os.chdir("..")

from copy import deepcopy
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure


# ## Outdoor Example
# 
# We recommend using our pre-trained model for input in outdoor environments because our model has only been trained on MegaDepth, and there exists a domain gap between indoor and outdoor data.

# In[4]:


from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

# You can choose model type in ['full', 'opt']
model_type = 'full' # 'full' for best quality, 'opt' for best efficiency

# You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
precision = 'fp32' # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

# You can also change the default values like thr. and npe (based on input image size)

if model_type == 'full':
    _default_cfg = deepcopy(full_default_cfg)
elif model_type == 'opt':
    _default_cfg = deepcopy(opt_default_cfg)
    
if precision == 'mp':
    _default_cfg['mp'] = True
elif precision == 'fp16':
    _default_cfg['half'] = True

print(_default_cfg)
matcher = LoFTR(config=_default_cfg)

matcher.load_state_dict(torch.load("EfficientLoFTR/weights/eloftr_outdoor.ckpt")['state_dict'])
matcher = reparameter(matcher) # no reparameterization will lead to low performance

if precision == 'fp16':
    matcher = matcher.half()
matcher = matcher.eval().cuda()

save_path = 'render_synthetic_far_matches/'  # TO_MODIFY -->decide where you want your results saved
folder_path = 'EfficientLoFTR/assets/r_sf_palestrina/'  # TO_MODIFY -->decide where is your dataset

with open(save_path+'matches.txt', 'w') as file:
	file.write("id     | matches\n")
images = os.listdir(folder_path)
images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# Sort the list of files by name
images.sort()
# Print the sorted list of files
print(images)
# Iterate over the files in the directory
for i in range(0, len(images), 2):
# Load example images
	img0_pth = folder_path+images[i]
	img1_pth = folder_path+images[i+1]
	img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
	img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
	img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))  # input size shuold be$
	img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))
	print("debug francesco 4")
	if precision == 'fp16':
		img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
		img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
	else:
		img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
		img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
	batch = {'image0': img0, 'image1': img1}

# Inference with EfficientLoFTR and get prediction
	with torch.no_grad():
		if precision == 'mp':
			with torch.autocast(enabled=True, device_type='gpu'):
				matcher(batch)
		else:
			matcher(batch)
	mkpts0 = batch['mkpts0_f'].cpu().numpy()
	mkpts1 = batch['mkpts1_f'].cpu().numpy()
	mconf = batch['mconf'].cpu().numpy()


# In[6]:


# Draw
	if model_type == 'opt':
		print(mconf.max())
		mconf = (mconf - min(20.0, mconf.min())) / (max(30.0, mconf.max()) - min(20.0, mconf.min()))

	color = cm.jet(mconf)
	text = [
			'eLoFTR',
			'Matches: {}'.format(len(mkpts0)),
			]
	fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, path=save_path+"match"+str(i)+".png")
# A png image will also be downloaded automatically.
	
	with open(save_path+'matches.txt', 'a') as file:
    		file.write(str(i)+"  |  "+str(len(mkpts0))+"\n")
print("END")
