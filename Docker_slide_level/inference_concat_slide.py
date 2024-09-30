import numpy as np
from const import *
import torch.nn.functional as F
from tqdm import tqdm
def concat_inference(outputs, coords, heatmap_height, heatmap_width, base_slide_mag):
    heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
    aux_heat_map = np.zeros((heatmap_height, heatmap_width), dtype=np.int8)
    
    coord_scale = base_slide_mag / TARGET_SLIDE_MAG
    for i, output in enumerate(tqdm(outputs)):


        output_sigmoid = output.numpy()
        patch_height, patch_width = output_sigmoid.shape[0], output_sigmoid.shape[1]

        left_x, left_y = round(coords[i][0] / coord_scale), round(coords[i][1] / coord_scale)

        end_y = min(left_y + patch_height, heatmap_height)
        end_x = min(left_x + patch_width, heatmap_width)
        result_tensor = output_sigmoid[:end_y-left_y, :end_x-left_x]
        try:
            heatmap[left_y:end_y, left_x:end_x] += result_tensor
            aux_heat_map[left_y:end_y, left_x:end_x] += 1

        except ValueError as e:
            print(f"Error at ({left_x}, {left_y}): {e}")
    
    heatmap2 = heatmap / aux_heat_map
    
    binary_heatmap = (heatmap2 >= 0.5).astype(np.uint8)
    return binary_heatmap