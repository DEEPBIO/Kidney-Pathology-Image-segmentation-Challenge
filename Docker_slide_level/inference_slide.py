import os
import argparse
import tifffile as tiff
from ensemble_model import load_model
from inference_patch_prep import prepare_patches
from inference_batch import batch_inference
from inference_concat_slide import concat_inference
import warnings
warnings.filterwarnings("ignore")

def load_slide_paths(data_dir):
    slide_paths = []
    for root,dirs,files in os.walk(data_dir):
        for file in files:
            if file.startswith("."):
                continue
            if file.endswith(".tiff") or file.endswith("tif"):
                slide_paths.append(os.path.join(root, file))
            else:
                continue
    return slide_paths

def inference(slide_path, model, args):
    #slide_folder = slide_path.split('/')[-2]
    #slide_name = slide_path.split('/')[-1].replace('wsi', 'mask')
    relative_path = os.path.relpath(slide_path, args.data_dir)
    mask_path = os.path.join(args.output_dir, relative_path)
    print(slide_path)
    patches, slide_width, slide_height, base_slide_mag = prepare_patches(slide_path=slide_path)
    
    outputs, coords = batch_inference(model, patches, batch_size=args.batch_size, device='cuda')
    binary_heatmap = concat_inference(outputs, coords, slide_height, slide_width, base_slide_mag)
    target_dir = os.path.dirname(mask_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    tiff.imwrite(mask_path, binary_heatmap*255)
    print(mask_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__) 
    parser.add_argument("--ckpt_folder", default=".",)
    parser.add_argument("--data_dir", default="/mnt/nfs7/workshop/val_wsi_level/")
    parser.add_argument("--output_dir", default='./output_dir')
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    model = load_model(args.ckpt_folder)
    slide_paths = load_slide_paths(args.data_dir)
    # slide_paths = ['/mnt/nfs7/workshop/val_wsi_level/56Nx/12-173_wsi.tiff', 
    #                '/mnt/nfs7/workshop/val_wsi_level/NEP25/18-577_wsi.tiff', 
    #                '/mnt/nfs7/workshop/val_wsi_level/DN/11-359_wsi.tiff',  
    #                '/mnt/nfs7/workshop/val_wsi_level/normal/normal_M2_wsi.tiff']
    for slide_path in slide_paths:
        inference(slide_path, model, args)
        
