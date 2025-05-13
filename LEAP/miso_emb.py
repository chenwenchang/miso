# basic import
import os
import re
from os.path import join
from tqdm import tqdm
import math
import ast
import random
import warnings


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import cv2
import openslide
from PIL import Image
import torch

# spatialdata import 
import geopandas as gpd
import anndata
import spatialdata as sd
import squidpy as sq
import scanpy as sc
from shapely.geometry import Point
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import (
    Affine,
    MapAxis,
    Scale,
    Sequence,
    Translation,
    get_transformation,
    set_transformation,
)

# MISO import 
import miso
from miso.hist_features import get_features




import argparse # <--- 添加这个导入
# --- 添加参数解析 ---
parser = argparse.ArgumentParser(description="Process a slice of subfolders.")
parser.add_argument("--start", type=int, required=True, help="Start index of the subfolder slice (inclusive).")
parser.add_argument("--end", type=int, required=True, help="End index of the subfolder slice (exclusive).")
cli_args = parser.parse_args()
# --- 结束参数解析 ---

# LOGGING
import logging
log_file_path = f'processing_log_{cli_args.start}_{cli_args.end}.log' # 例如 processing_log_0_20.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='w' # 每个进程写自己的日志文件
)

# Configuration
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch"
)  # Ignore specific torch warnings if desired
warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # Ignore potential future warnings from dependencies

# Seed for reproducibility
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Setup device (GPU if available, otherwise CPU)
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("CUDA is not available. Using CPU.")



def load_HE(path, level=3):
    slide = openslide.OpenSlide(path)
    image = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))[:, :, :3]
    return image

def get_img_path(path):
    """
    Get paths of different image in folder 'path'
    """
# Define matching rules in a dictionary.
    patterns = {
        'HnE': lambda base: base.lower().endswith('.svs'),
        'IMC': re.compile(r'_P3\.jpg$', re.IGNORECASE).search,
        'IMC3': re.compile(r'_P3\.jpg$', re.IGNORECASE).search,
        'IMC4': re.compile(r'_P4\.jpg$', re.IGNORECASE).search,
        'IMC5': re.compile(r'_P5\.jpg$', re.IGNORECASE).search,
        'IMC6': re.compile(r'_P6\.jpg$', re.IGNORECASE).search,
        'Visium': lambda base: base.lower().endswith('.tif'),
        'Anndata': lambda base: base.lower().endswith('.h5ad'),
    }
    
    # Initialize results to None.
    results = {key: [] for key in patterns.keys()}
    
    # Loop over the files, and for each file, check every pattern.
    files = sorted([f.path for f in os.scandir(path) if f.is_file()])
    for f in files:
        base = os.path.basename(f)
        for key, matcher in patterns.items():
            # If the matcher is callable (like our lambda or regex search), test it.
            if matcher(base):
                results[key].append(f)
                
    return results

    
def transform_corners(image_shape, H_matrix):
    h, w = image_shape[:2]

    # Define image corners in homogeneous coordinates
    corners = np.array(
        [
            [0, 0, 1],  # top-left
            [w - 1, 0, 1],  # top-right
            [w - 1, h - 1, 1],  # bottom-right
            [0, h - 1, 1],  # bottom-left
        ]
    ).T  # Shape: (3, 4)

    # Apply the homography
    transformed = H_matrix @ corners  # shape: (3, 4)

    # Normalize to Cartesian coordinates
    transformed /= transformed[2]

    # Return as (x, y) list
    return transformed[:2].T  # shape: (4, 2)

    
def process_subfolder(leap_id, files_path, H):
    try:
        adata = anndata.read_h5ad(file_in.get('Anndata')[0])
        centers = adata.obsm["spatial"]

        # HnE image
        print(f'[{leap_id}] Read H&E...')
        hne_img_yxc = load_HE(file_in['HnE'][0], level=1)
        hne_img = hne_img_yxc.transpose(2, 0, 1)
        img_for_sdata = Image2DModel.parse(
            data=hne_img,
            scale_factors=(2, 2, 2),
            c_coords=["r", "g", "b"]
        )

        # Mapping
        print(f'[{leap_id}] Mapping...')
        H = H_dict[leap_id]
        visium_h, visium_w = 3000, 3000
        flip = np.array([[-1, 0, visium_w - 1],
                         [ 0, 1, 0],
                         [ 0, 0, 1]])
        hne_scale = 4
        S = np.array([[hne_scale, 0, 0],
                      [0, hne_scale, 0],
                      [0, 0, 1]], dtype=float)

        H_all = S @ np.array(H) @ flip
        scale_factor = np.sqrt(abs(np.linalg.det(H_all[:2, :2])))
        S_ = np.array([[scale_factor, 0, 0],
                       [0, scale_factor, 0],
                       [0, 0, 1]], dtype=float)

        after_size = (
            round(visium_h * scale_factor),
            round(visium_w * scale_factor)
        )
        try:
            img_warp = cv2.warpPerspective(
                hne_img_yxc, 
                S_ @ np.linalg.inv(H_all), 
                after_size
            )
        except:
            corners = transform_corners((visium_h, visium_w), H_all)
            max_h, max_w = round(max(corners[:, 0])), round(max(corners[:, 1]))
            hne_h, hne_w = hne_img_yxc.shape[:2]
            print()
            hne_img_crop = hne_img_yxc[: min(max_h, hne_h), : min(max_w, hne_w)]
            img_warp = cv2.warpPerspective(hne_img_crop, S_ @ np.linalg.inv(H_all), after_size)
            
        # Update adata.uns
        spatial_key = "spatial"
        library_id = leap_id
        img_dict = adata.uns[spatial_key][library_id]["images"]
        scf = adata.uns[spatial_key][library_id]["scalefactors"]

        img_dict["lowres"] = img_dict["hires"]
        img_dict["hires"] = img_warp

        scf["tissue_lowres_scalef"] = scf["tissue_hires_scalef"]
        scf["tissue_hires_scalef"] = scale_factor

        # Build shapes GeoDataFrame
        px_per_um = 1 / 4.65
        radius = px_per_um * 55 / 2
        df = pd.DataFrame([radius] * len(centers), columns=["radius"])
        gdf = gpd.GeoDataFrame(df,
                               geometry=[Point(x, y) for x, y in centers])
        shapes_for_sdata = ShapesModel.parse(gdf)

        # Build spatialdata
        adata_for_sdata = TableModel.parse(adata)
        adata_for_sdata.uns["spatialdata_attrs"] = {
            "region": "spots",
            "region_key": "region",
            "instance_key": "spot_id",
        }
        adata.obs["region"] = pd.Categorical(["spots"] * len(adata))
        adata.obs["spot_id"] = shapes_for_sdata.index

        sdata = sd.SpatialData(
            images={"hne": img_for_sdata},
            shapes={"spots": shapes_for_sdata},
            tables={"adata": adata_for_sdata},
        )

        # Apply spatial transformations
        affine = Affine(H_all, input_axes=("x", "y"), output_axes=("x", "y"))
        # scale = Scale([4.0, 4.0], axes=("x", "y"))
        set_transformation(
            sdata.shapes["spots"], affine,
            to_coordinate_system="global"
        )
        sdata.shapes["spots_tf"] = sd.transform(
            sdata.shapes["spots"], maintain_positioning=True,
            to_coordinate_system="global"
        )

        # Extract embedding
        print(f'[{leap_id}] Calculate embedding...')
        highest_res_img_key = list(sdata.images["hne"].keys())[0]
        img_da = sdata.images["hne"][highest_res_img_key]["image"]
        img_np = img_da.data.compute().transpose(1, 2, 0)
        hne_pil = Image.fromarray(img_np)

        spots_pixel_gdf = sdata.shapes["spots_tf"]
        radius = spots_pixel_gdf["radius"].iloc[0]
        pixel_size_raw = pixel_size = 1

        ad = sdata.tables["adata"]
        locs_df = pd.DataFrame({
            "2": ad.obs["array_row"].astype(int).tolist(),
            "3": ad.obs["array_col"].astype(int).tolist(),
            "4": spots_pixel_gdf.geometry.y.round().astype(int).tolist(),
            "5": spots_pixel_gdf.geometry.x.round().astype(int).tolist()
        }, index=ad.obs_names)
        locs_df["1"] = ad.obs.get("in_tissue", pd.Series(1, index=ad.obs_names)).astype(int)
        # print(locs_df)
        image_emb = get_features(
            img=hne_pil,
            locs=locs_df,
            rad=radius,
            pixel_size_raw=pixel_size_raw,
            pixel_size=pixel_size,
            pretrained=True,
            device=device,
        )
        # deal with NaN
        if np.isnan(image_emb).any():
            image_emb = np.nan_to_num(image_emb, nan=0.0)
        adata.obsm["hne_emb"] = image_emb
        sdata.write(os.path.join(output_path, f"{leap_id}_sdata.zarr"), overwrite=True)

        # save images for fast checking
        # 1. Pre-calculate Neighbors and Leiden
        sc.pp.neighbors(adata, use_rep="hne_emb")
        sc.tl.leiden(adata, key_added="leiden_from_embedding")

        color_key1 = "leiden_stamp"
        color_key2 = "leiden_from_embedding"
        
        # 2. Create a figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 3. Generate each plot on its specific subplot axis
        
        # Plot 1: Spatial Scatter with 'leiden_stamp' (or fallback)
        sq.pl.spatial_scatter(
            adata,
            color=color_key1,
            ax=axes[0],      # Target the first subplot
            # show=False,    # REMOVE THIS LINE for sq.pl.spatial_scatter
            title=f"Spatial ({color_key1})"
        )

        # Plot 2: UMAP
        sc.pl.umap(
            adata,
            color=color_key2,
            ax=axes[1],      # Target the second subplot
            show=False,      # Keep this for sc.pl.umap (it handles it)
            title=f"UMAP ({color_key2})"
        )
        
        # Plot 3: Spatial Scatter with 'leiden'
        sq.pl.spatial_scatter(
            adata,
            color=color_key2,
            ax=axes[2],      # Target the third subplot
            # show=False,    # REMOVE THIS LINE for sq.pl.spatial_scatter
            title=f"Spatial ({color_key2})"
        )
        
        # 4. Adjust layout
        plt.tight_layout()
        
        # 5. Save the combined figure
        output_filename = os.path.join(output_path, f"{leap_id}_combined.jpg")
        # print(f"Saving combined plot to: {output_filename}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        
        # 6. Close the figure
        plt.close(fig)

    except Exception as e:
        return (leap_id, str(e))
    return None

    
if __name__ == '__main__':
    logging.info('Loading H matrics.')
    reg_excel = 'C:/Users/hua01/Desktop/cwc/Registration/spatial-registration/results/Registration_5.xlsx'
    reg_df = pd.read_excel(reg_excel)
    visium_df = reg_df[(reg_df['Rectangle'] == 'Y') & (reg_df['Registration_Type'] == 'Visium2HnE')]
    visium_df['leap'] = visium_df['Src_FileName'].str.split('_').str[0]
    visium_df['H'] = visium_df['Homography_Matrix'].apply(ast.literal_eval)
    H_dict = visium_df[['leap', 'H']].set_index('leap').to_dict()['H']
    flip_dict = visium_df[['leap', 'Src_Flip']].set_index('leap').to_dict()['Src_Flip']
    
    input_path = 'C:/Users/hua01/Desktop/cwc/Registration/Results_new/'
    output_path = 'C:/Users/hua01/Desktop/cwc/SpatialMultimodal/spd_new/'
    os.makedirs(output_path, exist_ok=True)

    logging.info('Start looping...')
    subfolders = sorted([f.path for f in os.scandir(input_path) if f.is_dir()])
    total_folders = len(subfolders)

    # --- 使用命令行参数进行切片 ---
    start_index = cli_args.start
    end_index = cli_args.end

    # 基本的边界检查
    if start_index < 0: start_index = 0
    if end_index > total_folders: end_index = total_folders
    if start_index >= end_index:
         print(f"Start index {start_index} is not before end index {end_index}. No folders to process.")
         logging.warning(f"Start index {start_index} is not before end index {end_index}. No folders to process.")
         exit() # 或者直接返回，取决于你的逻辑

    subfolders_to_process = subfolders[start_index:end_index]
    logging.info(f"Processing folders from index {start_index} to {end_index-1} (Total: {len(subfolders_to_process)} folders)")
    print(f"Processing folders from index {start_index} to {end_index-1} (Total: {len(subfolders_to_process)} folders)")
    # --- 结束切片 ---


    error_msg = []
    # --- 修改循环以遍历切片后的列表 ---
    # for subfolder in tqdm(subfolders[:50]): # 原来的循环
    for subfolder in tqdm(subfolders_to_process, desc=f"Processing {start_index}-{end_index}"): # 修改后的循环
        leap_id = os.path.basename(subfolder)
        logging.info(f'Start Processing {leap_id} ...')
        if leap_id not in H_dict:
            continue
            
        H = H_dict[leap_id]
        file_in = get_img_path(subfolder)
        err = process_subfolder(leap_id, file_in, H)
        if err is not None:
            print(f'{err[0]}: {err[1]}')
            error_msg.append(err)
        pd.DataFrame([x for x in error_msg if x is not None]).to_csv(f'err_msg_{start_index}_{end_index-1}.csv')