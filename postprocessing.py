import os
import glob
import tifffile as tiff
import numpy as np
import imagej
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from typing import List, Tuple, Iterator
from PIL import Image

import numpy as np
from skimage.measure import regionprops
from skimage.measure import label as sklabel
from typing import Tuple
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from skimage.transform import resize

from utilities import (intensity_flat_field_mask, 
                       oir_to_tif, 
                       match_channels, 
                       background_substruction, 
                       flat_field_correction, 
                       decomposition,
                       stiching_combinations,
                       find_decomp_files,
                       combine_channels,
                       find_stiching_map,
                       tiles_stitching)

# Add path
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-21.0.8.9-hotspot"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]

# create imagej session
ij = imagej.init('sc.fiji:fiji', mode="headless")

# input parameters
data_folder = r"\NoRI\Masha\20241120 Ageing Atlas 9mo"
stitched_files_folder = r"\NoRI\Masha\Stitched"
powersetting='UP'
# merge_order = ["NORI_c1", "NORI_c2", "NORI_c3"]#, "IF_c1", "IF_c2", "IF_c3", "IF_c4"]
file_separator = '_MAP'
overwrite_files = False
drive_letter = "Z:"
network_path = r"\\research.files.med.harvard.edu\Sysbio"
calibration_folder = os.path.join(drive_letter + r"\NoRI\Calibration Archive\20241120 calibration")

# Dependent varibles
rename_files_folder = 'signalX_test'
bg_files_folder = 'signal_bg_test'
ffc_files_folder = bg_files_folder + '_ffc2'
decomp_files_folder = 'decomp_bg_ffc2_test'
nori_channels_folder = os.path.join(decomp_files_folder, "composite")
VERBOSITY = 4
TIF_EXT = "*.tif"
OIR_EXT = "*.oir"
decomp_m_filename = 'M_bgffc2_linpol1.mat'
MAP_FOLDER_NAME_IDX = 4
CHANNEL_FOLDER_NAME_IDX = 2
OVERWRITE_IF_MAPS_READY = False
KEEP_SEPARATE_CHANNELS = False

# Data connection
mount_cmd = f'net use {drive_letter} {network_path}'
os.system(mount_cmd)
path = os.path.join(drive_letter + data_folder)
path_stitched = os.path.join(drive_letter + stitched_files_folder)

# Get folders list
folders = os.listdir(path)
folders = list(filter(lambda p: os.path.isdir(os.path.join(path, p)), folders))

# Calibration data
# Compute dark noize
bg_file_path = os.path.join(calibration_folder + r"\cal_linpol1\signalX\bg.tif")
bg_files = os.listdir(os.path.join(calibration_folder + r"\cal_linpol1\signalX"))
imbg = tiff.imread(bg_file_path)
bglevel = np.mean(imbg)

# Arrays to compute flat field correction
corr_mask_path = os.path.join(calibration_folder + r"\cal_linpol1\signal_bg")
corr_mask_files = os.listdir(corr_mask_path)
calculate_z_flag = int('maxz.mat' not in corr_mask_files)

# Spectral decomposition matrix
# Using linear unmixing, derive the protein/lipid/water fractions from
decomp_matrix = sio.loadmat(os.path.join(calibration_folder, decomp_m_filename)) # M
decomp_matrix_save = decomp_matrix.copy()
density_val = pd.DataFrame(decomp_matrix['M'][0])['density'].iloc[0]

if type(decomp_matrix)==dict:
    decomp_matrix = pd.DataFrame(decomp_matrix['M'][0])['M'].iloc[0]

# Normalisation
DECOMP_CONVERSION_FACTOR = 2**13
# volume fraction 'vv' or v/v' in ml/ml by concentration 'wv' or 'w/v' in g/ml
outputunit = 'vv'
normalization_option = "on"

if outputunit == 'vv':
    unitconversion = [1, 1, 1, 1]
elif outputunit == 'wv':
    unitconversion = [density_val[0][0], density_val[1][1], density_val[2][2]]
else:
    unitconversion = [1, 1, 1, 1]

# Read decomposition parameters used for calibration
with open (os.path.join(calibration_folder, 'decomp_data_bg.m')) as f:
    decomp_data = f.read()
channels = decomp_data.split('channelnames = {')[1].split('}\n')[0].replace("'", '').split(',')
channelnamesstr = [channels[2], channels[1], channels[0]]
cal_paths = decomp_data.split('samplenamestr = {')[1].split('}\n')[0].replace("'", '').split(',')

cal_lipidpath = cal_paths[2] + '_' + channelnamesstr[2] + '.tif'
cal_proteinpath = cal_paths[0] + '_' + channelnamesstr[1] + '.tif'
# for water used protein samples with water raman spectrum
cal_waterpath = cal_paths[0] + '_' + channelnamesstr[0] + '.tif'
strw, strp, strl = channelnamesstr

# Processing (2) load flat field correction mask
# Compute indexes for each channel
if calculate_z_flag:
    for file in corr_mask_files:
        if file.lower() == cal_lipidpath.lower():
            imout = tiff.imread(os.path.join(corr_mask_path, file))
            lipid_index = np.argmax(imout.mean(axis=1).mean(axis=1)) + 1
        if file.lower() == cal_proteinpath.lower():
            imout = tiff.imread(os.path.join(corr_mask_path, file))
            protein_index = np.argmax(imout.mean(axis=1).mean(axis=1)) + 1
        if file.lower() == cal_waterpath.lower():
            imout = tiff.imread(os.path.join(corr_mask_path, file))
            water_index = np.argmax(imout.mean(axis=1).mean(axis=1)) + 1
    
    sio.savemat(os.path.join(corr_mask_path, 'maxz.mat'), {
        'lipid_index': lipid_index,
        'protein_index': protein_index,
        'water_index': water_index
    })        
else:
    data = sio.loadmat(os.path.join(corr_mask_path, 'maxz.mat'))
    lipid_index = int(data['lipid_index'])
    protein_index = int(data['protein_index'])
    water_index = int(data['water_index'])

# Select masks with correct z possition
for file in corr_mask_files:
    if file.lower() == cal_lipidpath.lower():
        imout = tiff.imread(os.path.join(corr_mask_path, file))
        n_lipid = imout[lipid_index - 1]
    if file.lower() == cal_proteinpath.lower():
        imout = tiff.imread(os.path.join(corr_mask_path, file))
        n_protein = imout[protein_index - 1]
    if file.lower() == cal_waterpath.lower():
        imout = tiff.imread(os.path.join(corr_mask_path, file))
        n_water = imout[water_index - 1]

# Prepare masks for all possible resolutions
masks = {}
for size in [256, 512, 640, 800, 1024, 2048, 4096]:
    masks[f"protein_{size}"] = intensity_flat_field_mask(n_protein, size, size)
    masks[f"lipid_{size}"]   = intensity_flat_field_mask(n_lipid, size, size)
    masks[f"water_{size}"]   = intensity_flat_field_mask(n_water, size, size)

# Find all possible combinations of nori tiles
possible_stitching_combinations = stiching_combinations()

# Process all nori files inside data directory
for folder in folders[0:]:
    print(folder)
    # Folders for outputs
    os.makedirs(os.path.join(path, folder, rename_files_folder), exist_ok=True)
    os.makedirs(os.path.join(path, folder, bg_files_folder), exist_ok=True)
    os.makedirs(os.path.join(path, folder, ffc_files_folder), exist_ok=True)
    os.makedirs(os.path.join(path, folder, decomp_files_folder), exist_ok=True)
    composite_dir = os.path.join(path, folder, decomp_files_folder, 'composite')
    os.makedirs(composite_dir, exist_ok=True)

    all_if_files = []
    # Find all .oir files
    oir_files = glob.glob(os.path.join(os.path.join(path, folder), 
                                       "**", OIR_EXT), 
                                       recursive=True)
    for oir_file in oir_files:
        if 'Zone.Identifier' not in oir_file:
            file_name = oir_file.split('\\')[-1].split('.oir')[0]
            if file_name[:3]!='Map':
                # Convert oir files to tif
                tif_path = oir_file.replace('.oir', '.tif')
                image_np = oir_to_tif(oir_file, ij, tif_path)
                # check if IF file
                if '_IF_' in oir_file:
                    all_if_files.append(oir_file.replace('.oir', '.tif'))

                # Match nori channels
                if '_NORI_' in file_name:
                    immask_channel, renamed_file = match_channels(image_np, 
                                                                    oir_file, 
                                                                    file_name, 
                                                                    masks, 
                                                                    path, 
                                                                    folder, 
                                                                    rename_files_folder)
                
                    # Processing (1) background subtraction
                    if immask_channel is not None:
                        imout = background_substruction(image_np, 
                                                        bglevel, 
                                                        path, 
                                                        folder, 
                                                        bg_files_folder, 
                                                        renamed_file)
                                
                        if imout is not None:
                            # Processing (3) apply flat field correction mask
                            imffc = flat_field_correction(imout,
                                                            immask_channel, 
                                                            path, 
                                                            folder, 
                                                            ffc_files_folder, 
                                                            renamed_file)

    # Processing (4) decomposition
    all_flat_files = os.listdir(os.path.join(path, folder, ffc_files_folder))
    all_water_files = list(filter(lambda p: strw in p, all_flat_files))
    for water_file in all_water_files[0:]:
        decomposition(water_file, decomp_matrix,
                        unitconversion, DECOMP_CONVERSION_FACTOR,
                        strw, strl, strp,
                        path, folder, 
                        ffc_files_folder, decomp_files_folder,
                        normalization_option,
                        decomp_matrix_save,
                        outputunit)

    # Find all nori files for combination  
    all_decomp_files = os.listdir(os.path.join(path, folder, decomp_files_folder))
    samples = find_decomp_files(all_decomp_files,
                                file_separator)

    # Combine all files
    for sample_name in pd.unique(samples['sample_name']):
        df_name = samples[samples['sample_name']==sample_name]
        for map_name in pd.unique(df_name['map_name']):
            df_map = df_name[df_name['map_name']==map_name]
            tiles_number = df_map['tile_id'].max()
            poss_comb = possible_stitching_combinations[tiles_number]


            (file_names, 
             all_prot_images, 
             all_lipid_images, 
             all_water_images) = combine_channels(df_map, tiles_number, 
                                            strp, strl, strw, 
                                            path, folder, decomp_files_folder)

            # Compute constant shift
            tile_size = all_prot_images[0].shape
            tile_size = (3, tile_size[0], tile_size[1])
            shift = int(tile_size[1]*0.05)

            x, y, shift = find_stiching_map(all_prot_images, poss_comb, shift)
            print(sample_name, map_name, x, y, shift)

            # Stitch all tiles to one image
            tiles_stitching(all_if_files,
                            x, 
                            y, 
                            shift, 
                            path, 
                            folder, 
                            decomp_files_folder,
                            path_stitched,
                            file_separator,
                            tile_size)
            break
        break
    break

