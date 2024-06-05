Author: Bordignon Francesco 
Institution: University of Padua

TEST ELOFTR demo_mixed_pairs.py
1) To test eLoFTR with a folder of image pairs you can download eLoFTR from github in Tesi/ and follow their instructions to set up the environment
2) place the file demo_mixed_pairs.py in /EfficientLoFTR and create a folder for the resuts Tesi/result_folder containing a file matches.txt
3) place your folder of ordered pairs of images in Tesi/EfficientLoFTR/assets/
4) modify save_path (you shuld put your result_folder path) and dataset_path in demo_mixed_pairs.py
5) run the code in the eloftr environment on a CUDA device. The matches will be in the result_folder.
