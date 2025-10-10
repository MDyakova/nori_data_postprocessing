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