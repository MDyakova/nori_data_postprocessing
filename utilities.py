"""
Fuctions for main.py
"""

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


def crop_zoom(IM, zoom):
    """
    Crop center of mask
    """
    numrow1, numcol1 = IM.shape[:2]
    # Compute center
    r_center = numrow1 / 2
    c_center = numcol1 / 2
    # Compute half-size of cropped region (scaled by zoom)
    half_r = numrow1 / (2 * zoom)
    half_c = numcol1 / (2 * zoom)
    # Compute start and end indices (clip to array bounds)
    r1 = int(np.floor(r_center - half_r))
    r2 = int(np.ceil(r_center + half_r))
    c1 = int(np.floor(c_center - half_c))
    c2 = int(np.ceil(c_center + half_c))
    # Crop
    IM_crop = IM[r1:r2, c1:c2]
    return IM_crop

# import numpy as np
# from skimage.measure import regionprops

def parxyz(mask, phase):
    """
    Extracts (x, y, z) for each labeled region in the mask.
    x, y = centroid coordinates
    z = mean intensity in the region (from 'phase')
    """
    props = regionprops(mask, intensity_image=phase)
    
    x, y, z = [], [], []
    for p in props:
        if not np.isnan(p.mean_intensity):
            y.append(p.centroid[0])  # row coordinate
            x.append(p.centroid[1])  # column coordinate
            z.append(p.mean_intensity)

    return np.array(x), np.array(y), np.array(z)

def fit_poly_surface(x, y, z, deg_x=4, deg_y=3):
    """
    Compute polynomial surface
    """
    X = np.column_stack((x, y))
    
    poly = PolynomialFeatures(degree=max(deg_x, deg_y), include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression().fit(X_poly, z)
    return model, poly

def resize_image(image, width, height):
    """
    Resize image
    """
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized

def intensity_flat_field_mask(n_array, width, height):
    """
    Prepare flat fiels correction mask for protein, lipid and water (and dna, if presented)
    """
    # crop center
    ncol, nrow = n_array.shape
    zoom = 1
    im = crop_zoom(n_array, zoom)
    # divide image into blocks
    if nrow >= 512:
        step = 4
    else:
        step = 2
    mask = np.zeros_like(im, dtype='int')
    
    cc = 1
    for c in range(0, ncol - step + 1, step):
        for r in range(0, nrow - step + 1, step):
            mask[c:c+step, r:r+step] = cc
            cc += 1
    # compute mean intensity per region
    y, x, z = parxyz(mask, im)
    # fit smooth polynomial surface
    model, poly = fit_poly_surface(x, y, z, deg_x=3, deg_y=4)
    
    # Predict new values
    X_pred = np.column_stack((x, y))   # flattened grid
    Z_pred = model.predict(poly.transform(X_pred))
    Bg = Z_pred.reshape((int(Z_pred.shape[0]**0.5), int(Z_pred.shape[0]**0.5)))
    # normalize to produce mask
    im_mask = Bg / Bg.max()
    im_mask = resize(im_mask, (ncol, nrow), order=1, mode='reflect', anti_aliasing=True)

    # Resize for non-standart resolution
    im_mask = resize_image(im_mask, width, height)
    return im_mask

def oir_to_tif(oir_file, ij, tif_path):
    """
    Convert oir format to tif
    """
    dataset = ij.io().open(oir_file)
    image_np = ij.py.from_java(dataset)
    image_np = image_np.astype('uint32')
    tiff.imwrite(tif_path, image_np.astype('uint32'))
    return image_np

def match_channels(image_np, oir_file, file_name, masks, path, folder, rename_files_folder):
    """
    Match nori channels and return flat fiels correction mask
    """
    if '_Cycle_02\\' in oir_file:
        renamed_file = file_name + '_channel_waterUP.tif'
        immask_channel = masks[f'water_{str(image_np.values.shape[0])}']
    elif '_Cycle_01\\' in oir_file:
        renamed_file = file_name + '_channel_lipidUP.tif'
        immask_channel = masks[f'lipid_{str(image_np.values.shape[0])}']
    elif '_Cycle\\' in oir_file:
        renamed_file = file_name + '_channel_proteinUP.tif'
        immask_channel = masks[f'protein_{str(image_np.values.shape[0])}']
    else:
        immask_channel = None

    file_path = os.path.join(path, folder, rename_files_folder, renamed_file)
    tiff.imwrite(file_path, image_np.astype('float32'))

    return immask_channel, renamed_file

def background_substruction(image_np, 
                            bglevel, 
                            path, 
                            folder, 
                            bg_files_folder, 
                            renamed_file):
    if '_channel_confocal.tif' not in renamed_file:
        imout = image_np.values - bglevel
        tiff.imwrite(os.path.join(path, folder, bg_files_folder, renamed_file), imout.astype('float32'))
        return imout
    else:
        return None
    
def flat_field_correction(imout,
                          immask_channel, 
                          path, 
                          folder, 
                          ffc_files_folder, 
                          renamed_file):
    if len(imout.shape)==2:
        if imout.shape[0]==imout.shape[1]:
            # if imout.shape == immask_channel.shape:
            imffc = imout / immask_channel
            # else:
            #     resized = cv2.resize(immask_channel, imout.shape, interpolation=cv2.INTER_LINEAR)
            #     imffc = imout / resized
        else:
            return None
    else:
        if imout.shape[1]==imout.shape[2]:
            mask = np.repeat(immask_channel[:, :, np.newaxis], imout.shape[2], axis=2)
            imffc = imout / mask
        else:
            return None
    tiff.imwrite(os.path.join(path, folder, ffc_files_folder, renamed_file), imffc.astype('float32'))
    
    return 

def decomposition(water_file, decomp_matrix,
                  unitconversion, DECOMP_CONVERSION_FACTOR,
                  strw, strl, strp,
                  path, folder, 
                  ffc_files_folder, decomp_files_folder,
                  normalization_option,
                  decomp_matrix_save,
                  outputunit):
    if 'Zone.Identifier' not in water_file:
        prefix = water_file.split(strw)[0]
        lipid_file = prefix + strl + '.tif'
        protein_file = prefix + strp + '.tif'
    
        protein_im = tiff.imread(os.path.join(path, folder, ffc_files_folder, protein_file)).astype('float32')
        lipid_im = tiff.imread(os.path.join(path, folder, ffc_files_folder, lipid_file)).astype('float32')
        water_im = tiff.imread(os.path.join(path, folder, ffc_files_folder, water_file)).astype('float32')
        data = np.vstack([
            lipid_im.ravel(),
            protein_im.ravel(),
            water_im.ravel()
        ])
        decomp_output = np.matmul(decomp_matrix, data)
        decomp_output = np.where(decomp_output<0, 0, decomp_output)
        
        if normalization_option.lower() == "on":
            total = np.sum(decomp_output[0:3, :], axis=0) + 1e-6
        else:
            total = 100
    
        decomp_l = unitconversion[0] * np.reshape(
            decomp_output[0, :] / total, 
            lipid_im.shape
        )
        decomp_p = unitconversion[1] * np.reshape(
            decomp_output[1, :] / total, 
            protein_im.shape
        )
        decomp_w = unitconversion[2] * np.reshape(
            decomp_output[2, :] / total, 
            water_im.shape
        )
        decomp_p = (DECOMP_CONVERSION_FACTOR*decomp_p).astype('float32')
        decomp_l = (DECOMP_CONVERSION_FACTOR*decomp_l).astype('float32')
        decomp_w = (DECOMP_CONVERSION_FACTOR*decomp_w).astype('float32')

        # decomp_p = np.where(decomp_p<0, 0, decomp_p)
        # decomp_l = np.where(decomp_l<0, 0, decomp_l)
        # decomp_w = np.where(decomp_w<0, 0, decomp_w)
    
        tiff.imwrite(os.path.join(path, folder, decomp_files_folder, protein_file), decomp_p)
        tiff.imwrite(os.path.join(path, folder, decomp_files_folder, lipid_file), decomp_l)
        tiff.imwrite(os.path.join(path, folder, decomp_files_folder, water_file), decomp_w)
        # if decomp_output.shape[0] == 4:
        #     decomp_w = unitconversion[2] * np.reshape(
        #         decomp_output[3, :] / total, 
        #         water_im.shape
        #     )
        #     decomp_m = (DECOMP_CONVERSION_FACTOR*decomp_m).astype('float32')
        # else:
        #     decomp_m = np.NaN
    
        # # nosignal = ((lipid_im + protein_im + water_im) == 0)
        
        sio.savemat(os.path.join(path, folder, decomp_files_folder, 'M.mat'), {"M": decomp_matrix_save})   
        log_path = os.path.join(path, folder, decomp_files_folder, "note.txt")
        
        with open(log_path, "w") as f:
            f.write(f"Normalization is '{normalization_option}'.\n")
            f.write(f"Output format is '{outputunit}'.\n")
            f.write(f"Tiff image conversion factor is {DECOMP_CONVERSION_FACTOR}.\n")
            
def stiching_combinations():
    possible_stitching_combinations = {}
    for n in range(1, 101):
        combos = []
        for rows in range(1, n + 1):
            if n % rows == 0:
                cols = n // rows
                combos.append([rows, cols])
        combos_sorted = sorted(combos, key=lambda rc: abs(rc[0] - rc[1]))
        possible_stitching_combinations[n] = combos_sorted
    return possible_stitching_combinations
