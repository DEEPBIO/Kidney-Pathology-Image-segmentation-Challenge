from torchvision import transforms
from openslide import OpenSlide
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
from const import *
def process_patch(slide, w, h, scale, transform, INFERENCE_INPUT_SIZE):
    patch = slide.read_region((w, h), 0, (INFERENCE_INPUT_SIZE*scale, INFERENCE_INPUT_SIZE*scale)).convert('RGB')
    return (transform(patch), (w, h))

def prepare_patches(slide_path):
    transform = transforms.Compose([
        transforms.Resize((INFERENCE_INPUT_SIZE, INFERENCE_INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    patches = []
    if "NEP25" in slide_path:
        BASE_SLIDE_MAG = 200
    else:
        BASE_SLIDE_MAG = 400
    scale = int(BASE_SLIDE_MAG / INFERENCE_INPUT_MAG) # There is only exists 400 mag. We need to resize it

    slide = OpenSlide(slide_path)
    slide_width, slide_height = slide.dimensions
    
    
    process_partial = partial(process_patch, slide, scale=scale, transform=transform, INFERENCE_INPUT_SIZE=INFERENCE_INPUT_SIZE)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for w in range(0, slide_width - INFERENCE_INPUT_STRIDE * scale + 1, INFERENCE_INPUT_STRIDE * scale):
            for h in range(0, slide_height - INFERENCE_INPUT_STRIDE * scale + 1, INFERENCE_INPUT_STRIDE * scale):
                futures.append(executor.submit(process_partial, w, h))
        
        for future in tqdm(futures):
            patches.append(future.result())
    
    slide_width = round(slide_width / round(BASE_SLIDE_MAG / TARGET_SLIDE_MAG))
    slide_height = round(slide_height / round(BASE_SLIDE_MAG / TARGET_SLIDE_MAG))
    return patches, slide_width, slide_height, BASE_SLIDE_MAG



if __name__ == "__main__":
        
    slide_path = "/mnt/nfs7/workshop/val_wsi_level/NEP25/18-575_wsi.tiff"    
    mask_path = "/mnt/nfs7/workshop/val_wsi_level/NEP25/18-575_mask.tiff"
    patches, slide_width, slide_height = prepare_patches(slide_path=slide_path)
    
    import ipdb; ipdb.set_trace()
