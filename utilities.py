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
from scipy.ndimage import shift as imshift

# Dict to correct flourescence shift error
fluorescence_shift_dict = {}
fluorescence_shift_dict[512] = (7, -4)
fluorescence_shift_dict[1024] = (29, -26)

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
    """
    Remove background
    """
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
    """
    Make flat field correction
    """
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
    """
    Make NORI images decomposition
    """
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
    """
    Return all possible combinations for nori tiles
    """
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

def find_decomp_files(all_decomp_files,
                      file_separator):
    """
    Return all files for decomposition
    """
    samples = []
    for file in all_decomp_files:
        if ('.tif' in file) & ('Zone.Identifier' not in file):
            sample_name = file.split(file_separator)[0]
            map_name = file.split(file_separator)[1].split('_')[0]
            tile_id = int(file.split('_channel')[0].split('_')[-1])
            samples.append([file, sample_name, map_name, tile_id])
    samples = pd.DataFrame(samples, columns=('file', 'sample_name', 'map_name', 'tile_id'))
    return samples

def clip_inplace(a, lo, hi):
    # Clip but avoid NaNs
    np.clip(a, lo, hi, out=a)
    return a

def save_ome_tif(path, channels, axes="CZYX"):
    """
    channels: list of (Z,Y,X) float32 arrays in calibrated units.
    Saves as float32 OME-TIFF with given axes.
    """
    data = np.stack(channels, axis=0)  # (C,Z,Y,X)
    metadata = {'axes': axes}
    tiff.imwrite(
        str(path),
        data,
        dtype=np.float32,
        photometric='minisblack',
        metadata=metadata,
        imagej=False
    )

def to_uint8_rgb(protein, lipid, water):
    """
    Inputs are float (Z,Y,X) in calibrated 0..1000 units after clipping.
    Convert each to 0..255 and stack as RGB per slice.
    Returns array (Z, Y, X, 3) uint8.
    """
    def scale01(x):
        # Map 0..1000 → 0..1 (consistent with macro’s calibrated units)
        y = np.clip(x / 1000.0, 0, 1)
        return y

    R = (scale01(protein) * 255.0).astype(np.uint8)
    G = (scale01(lipid)   * 255.0).astype(np.uint8)
    B = (scale01(water)   * 255.0).astype(np.uint8)
    return np.stack([R, G, B], axis=-1)

def combine_channels(df_map, tiles_number, 
                     strp, strl, strw, 
                     path, folder, decomp_files_folder):
    """
    Function combines all channels images together
    """
    all_prot_images = []
    all_lipid_images = []
    all_water_images = []
    file_names = []
    for tile_id in range(1, tiles_number+1):
        tile_files = list(df_map[df_map['tile_id']==tile_id]['file'])
        protein_file = list(filter(lambda p: strp in p, tile_files))[0]
        protein_image = tiff.imread(os.path.join(path, folder, decomp_files_folder, protein_file))
        # all_prot_images.append(protein_image)
        # file_names.append(protein_file)

        lipid_file = list(filter(lambda p: strl in p, tile_files))[0]
        lipid_image = tiff.imread(os.path.join(path, folder, decomp_files_folder, lipid_file))
        # all_lipid_images.append(lipid_image)

        water_file = list(filter(lambda p: strw in p, tile_files))[0]
        water_image = tiff.imread(os.path.join(path, folder, decomp_files_folder, water_file))
        # all_water_images.append(water_image)
    
        protein_image = protein_image.astype(np.float32, copy=False)
        lipid_image   = lipid_image.astype(np.float32, copy=False)
        water_image   = water_image.astype(np.float32, copy=False)

        file_names.append(protein_file)
        all_prot_images.append(protein_image)
        all_lipid_images.append(lipid_image)
        all_water_images.append(water_image)

        out_drawing  = os.path.join(path, 
                                    folder, 
                                    decomp_files_folder, 
                                    'composite', 
                                    water_file.split('_channel')[0] + '_drawing.tif')
        out_composite   = os.path.join(path, 
                                       folder, 
                                       decomp_files_folder, 
                                       'composite', 
                                       water_file.split('_channel')[0] + '.tif')
        
        if not (water_image.shape == protein_image.shape == lipid_image.shape):
            raise ValueError(f"Shape mismatch for file {protein_file} in {folder}")
        
        channels = [protein_image, lipid_image, water_image]
        if len(protein_image.shape)==2:
            axes = "CYX" # if 2D
        else:
            axes = "CZYX" # if 3D
        
        # Save multi-channel composite as OME-TIFF (C,Z,Y,X), float32, units ~0..1000
        save_ome_tif(out_composite, channels, axes=axes)
        # Save RGB drawing (protein→R, lipid→G, water→B)
        rgb = to_uint8_rgb(protein_image, lipid_image, water_image)  # (Z,Y,X,3)
        # If multiple Z-slices, write an ImageJ-compatible stack of RGB pages
        # tifffile will write one page per Z with SamplesPerPixel=3
        tiff.imwrite(str(out_drawing), rgb, photometric='rgb')

    return file_names, all_prot_images, all_lipid_images, all_water_images

def snake_by_rows_indices(n_cols: int, n_rows: int,
                          start: str = "Right", vertical: str = "Down") -> List[int]:
    """Return flat indices (row-major) in snake-by-rows order."""
    if start not in {"Right", "Left"} or vertical not in {"Down", "Up"}:
        raise ValueError("start must be 'Right' or 'Left'; vertical must be 'Down' or 'Up'.")

    row_iter = range(n_rows) if vertical == "Down" else range(n_rows - 1, -1, -1)
    order: List[int] = []

    for i, r in enumerate(row_iter):
        left_to_right = (i % 2 == 0) if start == "Right" else (i % 2 == 1)
        col_iter = range(n_cols) if left_to_right else range(n_cols - 1, -1, -1)
        for c in col_iter:
            order.append(r * n_cols + c)
    return order

def step_through_images(files: List[str], n_cols: int, n_rows: int,
                        start: str = "Right", vertical: str = "Down",
                        pause: bool = False) -> Iterator[Tuple[int, int, int, str]]:
    """
    Yield (flat_idx, row, col, path) for each image in the specified order.
    Set pause=True to wait for Enter between steps.
    """
    if len(files) != n_cols * n_rows:
        raise ValueError(f"Expected {n_cols*n_rows} files, got {len(files)}.")

    order = snake_by_rows_indices(n_cols, n_rows, start, vertical)
    for idx in order:
        r, c = divmod(idx, n_cols)
        yield idx, r, c, files[idx]
        if pause:
            input("Press Enter for next...")

def find_stiching_map(all_prot_images, poss_comb, shift):
    """
    Check possible stiching combination and find the best parameters
    """
    res = []
    for comb in poss_comb:
        cols, rows = comb
        prev_image = []
        tile_size = all_prot_images[0].shape
        tile_size = (3, tile_size[0], tile_size[1])
        for step, (idx, r, c, next_im) in enumerate(
                step_through_images(all_prot_images, cols, rows, start="Right", vertical="Down", pause=False), 1):
            if len(prev_image)==0:
                prev_image = all_prot_images[step-1]
                prev_r = r
                prev_c = c
            else:
                new_image = all_prot_images[step-1]
                if prev_r==r:
                    if prev_c<c:
                        overlap_first = prev_image[:, -shift:]
                        overlap_second = new_image[:, :shift]
                        dist = np.corrcoef(overlap_first.reshape(-1), overlap_second.reshape(-1)).min()
                    else:
                        overlap_first = prev_image[:, :shift]
                        overlap_second = new_image[:, -shift:]
                        dist = np.corrcoef(overlap_first.reshape(-1), overlap_second.reshape(-1)).min()
                else:
                    overlap_first = prev_image[-shift:]
                    overlap_second = new_image[:shift]
                    dist = np.corrcoef(overlap_first.reshape(-1), overlap_second.reshape(-1)).min()
        
                res.append([comb[0], comb[1], dist, shift])
                
                new_image = all_prot_images[step-1]
                if prev_r==r:
                    if prev_c<c:
                        overlap_first = prev_image[:, -shift-1:]
                        overlap_second = new_image[:, :shift+1]
                        dist = np.corrcoef(overlap_first.reshape(-1), overlap_second.reshape(-1)).min()
                    else:
                        overlap_first = prev_image[:, :shift+1]
                        overlap_second = new_image[:, -shift-1:]
                        dist = np.corrcoef(overlap_first.reshape(-1), overlap_second.reshape(-1)).min()
                else:
                    overlap_first = prev_image[-shift-1:]
                    overlap_second = new_image[:shift+1]
                    dist = np.corrcoef(overlap_first.reshape(-1), overlap_second.reshape(-1)).min()
                
                res.append([comb[0], comb[1], dist, shift+1])
        
                prev_image = all_prot_images[step-1]
                prev_r = r
                prev_c = c
    res = pd.DataFrame(res, columns=('x', 'y', 'dist', 'shift'))
    res = res.groupby(by=['x', 'y', 'shift'], as_index=False).mean()
    res = res.sort_values(by=['dist'], ascending=False).iloc[0:1]
    x = res['x'].max()
    y = res['y'].max()  
    shift = res['shift'].max()
    return x, y, shift

import numpy as np
import cv2

def blend_distance_feather(img1, img2, mask1=None, mask2=None, eps=1e-6, power=1.0):
    """
    Realistic feathering using distance transforms.
    - Weights increase with distance from each tile's boundary.
    - 'power' > 1 sharpens the transition; < 1 softens it.

    img1, img2: HxW or HxWxC, same dtype/shape.
    mask1, mask2: uint8/bool masks of valid data (1/True where valid). If None, uses >0.

    Returns: blended image (same dtype).
    """
    f32 = np.float32
    a = np.asarray(img1, dtype=f32)
    b = np.asarray(img2, dtype=f32)
    H, W = a.shape[:2]

    if mask1 is None:
        mask1 = (a > 0).any(axis=-1) if a.ndim == 3 else (a > 0)
    if mask2 is None:
        mask2 = (b > 0).any(axis=-1) if b.ndim == 3 else (b > 0)

    # Distance to the nearest zero (edge) *inside* the valid region
    d1 = cv2.distanceTransform((mask1.astype(np.uint8))*255, cv2.DIST_L2, 3)
    d2 = cv2.distanceTransform((mask2.astype(np.uint8))*255, cv2.DIST_L2, 3)

    # Only consider places where at least one image is valid
    union = (mask1 | mask2)
    d1 = d1 ** power
    d2 = d2 ** power
    w1 = d1 / (d1 + d2 + eps)
    w2 = 1.0 - w1

    # Where only one is valid, make that weight 1
    w1[mask1 & ~mask2] = 1.0
    w2[mask1 & ~mask2] = 0.0
    w1[~mask1 & mask2] = 0.0
    w2[~mask1 & mask2] = 1.0
    w1[~union] = 0.0
    w2[~union] = 0.0

    if a.ndim == 3:
        w1 = w1[..., None]; w2 = w2[..., None]

    out = a * w1 + b * w2
    return out.astype(img1.dtype)

def tiles_stitching(all_if_files,
                    x, 
                    y, 
                    shift, 
                    path, 
                    folder, 
                    decomp_files_folder, 
                    path_stitched, 
                    file_separator,
                    tile_size):
    """
    Stitch all tiles to one image
    """
    max_x = x
    max_y = y

    coords_dict = {}
    for j in range(y):
        for i in range(x):
            if (i==0) & (j==0):
                coords_dict[(j, i)] = [0, 0]
                prev_x = i
                prev_y = j
            else:
                if i==0:
                    shift_x = 0
                    shift_y = tile_size[1] - shift
                else:
                    shift_x = tile_size[2] - shift
                    shift_y = 0 
                prev_coords = coords_dict[(prev_y, prev_x)]
                new_coord = [int(prev_coords[0]+shift_y), int(prev_coords[1]+shift_x)]
                coords_dict[(j, i)] = new_coord
                prev_x = i
                prev_y = j

                if prev_x==(max_x-1):
                    prev_y = j
                    prev_x = 0               

    shape_x = np.array(list(coords_dict.values())).T[1].max() + tile_size[2]
    shape_y = np.array(list(coords_dict.values())).T[0].max() + tile_size[1]

    composite_dir = os.path.join(path, folder, decomp_files_folder, 'composite')
    composite_files = os.listdir(composite_dir)
    composite_files = list(filter(lambda p: ('_drawing' not in p) & ('.tif' in p), composite_files))

    samples = []
    for file in composite_files:
        if '.tif' in file:
            sample_name = file.split(file_separator)[0]
            map_name = file.split(file_separator)[1].split('_')[0]
            tile_id = int(file.split('.tif')[0].split('_')[-1])
            samples.append([file, sample_name, map_name, tile_id])
    samples = pd.DataFrame(samples, columns=('file', 'sample_name', 'map_name', 'tile_id'))

    df_name = samples[samples['sample_name']==sample_name]
    df_map = df_name[df_name['map_name']==map_name]
    tiles_number = df_map['tile_id'].max()

    all_tile_files = []
    file_names = []
    for tile_id in range(1, tiles_number+1):
        tile_file = list(df_map[df_map['tile_id']==tile_id]['file'])[0]
        tile_image = tiff.imread(os.path.join(composite_dir, tile_file))
        all_tile_files.append(tile_image)
        file_names.append(tile_file)

    tile_size = all_tile_files[0].shape
    cols, rows = x, y
    new_image = np.zeros((tile_size[0], shape_y, shape_x))
    for step, (_, r, c, next_im) in enumerate(
            step_through_images(all_tile_files, cols, rows, start="Right", vertical="Down", pause=False), 1):
        next_im = all_tile_files[step-1]
        coords = coords_dict[(r, c)]
        # image_crop = new_image[:, coords[0]:coords[0]+tile_size[1], coords[1]:coords[1]+tile_size[2]].copy()
        new_image_mask = np.zeros((shape_y, shape_x))

        for layer in range(tile_size[0]):
            new_image_mask[coords[0]:coords[0]+tile_size[1], coords[1]:coords[1]+tile_size[2]] = next_im[layer]
            new_image[layer] = blend_distance_feather(new_image[layer], new_image_mask, eps=1e-6, power=0.5)
    out_stitched  = os.path.join(path_stitched, sample_name + '_MAP' + map_name + '.tif')

    # Fluorescent files stitching
    print(all_if_files)
    if len(all_if_files)>0:
        samples_if = []
        for file in all_if_files:
            if '.tif' in file:
                # path_if = '\\'.join(file.split('\\')[:-1])
                file_name = file.split('\\')[-1]
                sample_name = file_name.split(file_separator)[0]
                map_name = file_name.split(file_separator)[1].split('_')[0]
                tile_id = int(file_name.split('.tif')[0].split('_')[-1])
                samples_if.append([file, file_name, sample_name, map_name, tile_id])
        samples_if = pd.DataFrame(samples_if, columns=('file', 'file_name', 'sample_name', 'map_name', 'tile_id'))

        df_name = samples_if[samples_if['sample_name']==sample_name]
        df_map = df_name[df_name['map_name']==map_name]
        print(df_map['tile_id'].max())
        df_map.to_csv('test.csv')
        tiles_number = int(df_map['tile_id'].max())

        all_tile_if_files = []
        file_names_if = []
        for tile_id in range(1, tiles_number+1):
            tile_file = list(df_map[df_map['tile_id']==tile_id]['file'])[0]
            tile_image = tiff.imread(tile_file)
            all_tile_if_files.append(tile_image)
            file_names_if.append(tile_file)

        new_image_if = np.zeros((all_tile_if_files[0].shape[2], shape_y, shape_x))
        for step, (_, r, c, next_im) in enumerate(
                step_through_images(all_tile_if_files, cols, rows, start="Right", vertical="Down", pause=False), 1):
            next_im = all_tile_if_files[step-1]
            coords = coords_dict[(r, c)]
            # image_crop = new_image_if[:, coords[0]:coords[0]+tile_size[1], coords[1]:coords[1]+tile_size[2]].copy()
            new_image_mask_if = np.zeros((shape_y, shape_x))

            for layer in range(all_tile_if_files[0].shape[2]):
                new_image_mask_if[coords[0]:coords[0]+tile_size[1], coords[1]:coords[1]+tile_size[2]] = next_im[:, :, layer]
                new_image_if[layer] = blend_distance_feather(new_image_if[layer], new_image_mask_if, eps=1e-6, power=0.5)

        # Correct flourescence shift
        f_shift = fluorescence_shift_dict[tile_size[1]]
        shifted_images = []
        for layer in range(new_image_if.shape[0]):
            b_aligned = imshift(new_image_if[layer], shift=f_shift, order=1, mode='constant', cval=0.0)
            shifted_images.append(b_aligned)
        all_images = np.concatenate([new_image, shifted_images], axis=0)
        # Save as ImageJ-compatible multi-channel NORI + IF TIFF
        tiff.imwrite(
            out_stitched,
            all_images.astype('float32'),
            imagej=True,
            metadata={'axes': 'CYX'} 
        )
    else:
        # Save as ImageJ-compatible multi-channel NORI TIFF
        tiff.imwrite(
            out_stitched,
            new_image.astype('float32'),
            imagej=True,
            metadata={'axes': 'CYX'}
        ) 
