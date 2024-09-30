#!/usr/bin/env python3
import os
import datetime
import random
import argparse

import PIL
from PIL import Image
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from model import load_model
import warnings
warnings.filterwarnings('ignore')

DATA_SIZE = 2048
DATA_MAG = 200
THRESHOLD = 0.5
#MASK_DOWNSAMPLE = 4
#self.patch_size_in_image = 1024
#IMAGE_DOWNSAMPLE = 2

class Dataset_infer_patch(torch.utils.data.Dataset):

    def __init__(self,
                    PatchDir='/mnt/nfs7/workshop/patch_level',
                    patch_size=512,
                    mag=100,
                    mask_downsample=4, # 4 in general, 8 for EfficientViT
                    Train = True,
                    PatchTransform=None,
                    RotateFlipTransform=True,
                    RandomCrop=True,
                    RandomMpp=True,
                    CenterCrop=False,
                    debug = False):

        

        self._patch = PatchDir
        self._train = Train

        self.patch_size = patch_size
        self.mag = mag 
        self.image_downsample = DATA_MAG/self.mag # 2 if 100x 
        self.patch_size_in_image = round(self.patch_size*self.image_downsample)
        self.mask_downsample = mask_downsample


        self._patchtrans = PatchTransform
        self._rotfliptrans = RotateFlipTransform
        self._randomcrop = RandomCrop
        self._centercrop = CenterCrop 
        self._randommpp = RandomMpp
        self._totensor = torchvision.transforms.ToTensor()
        # Create class and data list.
        self._target_list = []
        self._data_list = []
    
        # /mnt/nfs7/workshop/patch_level/trn_data/56Nx/12_116/img/56Nx_12_116_4_4096_0_img.jpg
        # /mnt/nfs7/workshop/patch_level/trn_data/56Nx/12_116/mask/56Nx_12_116_4_4096_0_mask.jpg
        image_extensions = {'.jpg', '.jpeg', '.png',}  # 이미지 파일 확장자 집합
        # 주어진 디렉토리부터 시작하여 모든 하위 디렉토리를 순회
        for root, dirs, files in os.walk(PatchDir):
            for file in files:
                # 파일 확장자 확인
                if os.path.splitext(file)[1].lower() in image_extensions:
                    # 파일 경로를 리스트에 추가
                    self._data_list.append(os.path.join(root, file))
                    if debug and len(self._data_list) >= 100 : 
                        print(f'Number of data: {len(self)}')
                        return 
        
        print(f'Number of data: {len(self)}')

    def __len__(self):
        """Return the number of the patches to be .
        """
        return len(self._data_list)

    def __getitem__(self, index):
    
        patch_path = self._data_list[index]
        patch = PIL.Image.open(patch_path).convert('RGB')

        patch = patch.resize((self.patch_size, self.patch_size), PIL.Image.BILINEAR) 

        if self._patchtrans:
            patch = self._patchtrans(patch)

        patch = self._totensor(patch)

        return patch, patch_path

class ensemble_model(nn.Module) : 
    def __init__(self, models, threshold = THRESHOLD, scale_factor=8) : 
        super().__init__() 
        self.models = nn.ModuleList(models)
        self.threshold = threshold
        self.scale_factor=scale_factor
    
    def forward(self, x) : 
        outputs = torch.stack([model(x) for model in self.models], dim=0) # N, B, C, H, W
        N,B,C,H,W = outputs.shape
        assert C == 1 
        outputs = torch.nn.functional.interpolate(outputs.squeeze(2), size=(H*self.scale_factor, W*self.scale_factor), mode='bilinear')
        outputs = torch.nn.functional.sigmoid(outputs)
        #outputs = outputs.mean(0)
        #outputs = (outputs>self.threshold).type(torch.float)
        outputs = (outputs>self.threshold).type(torch.float)
        outputs = outputs.mean(0)
        outputs = (outputs>0.5).type(torch.float)
        return outputs
    
def inference(args):
    use_cuda = torch.cuda.is_available()
    # Prepare data.
    print("Preparing Data..")
    val_loader = DataLoader(Dataset_infer_patch(PatchDir=args.data_dir, Train=False, debug=args.debug_mode,
                                    patch_size=args.patch_size, mag=args.mag, mask_downsample=args.mask_downsample),
                            batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers,
                            shuffle=True, drop_last=False)


    # Prepare model.
    print("Preparing Model..")
     
    torch.cuda.empty_cache()

    model_list = [] 
    for filename in os.listdir('.') : 
        if filename.endswith('pth.tar'): 
            net = load_model(args.model_name, args.class_num)
            checkpoint = torch.load(filename)
            result = net.load_state_dict(checkpoint["model_state"], strict=True)
            model_list.append(net)
            print(f'Checkpoint loaded from {filename}.')
    net = ensemble_model(model_list)
    params = list(net.parameters())

    n_params = 0 
    for param in params : 
        n_params += torch.numel(param)
    print(f'Number of parameters: {n_params}')
    

    if use_cuda:
        net.cuda()
        cudnn.benchmark = True
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    # Run training.
    print("Start!")

    net.eval()

    with torch.autograd.no_grad():
        for idx, (inputs, filepaths) in enumerate(val_loader) : 
            
            # Wrap data.
            inputs = inputs.cuda()

            # Forward pass.
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = args.half_precision): 
                masks = net(inputs)
            
            for mask, filepath in zip(masks, filepaths) : 
                relative_path = os.path.relpath(filepath, args.data_dir)
                mask_path = os.path.join(args.output_dir, relative_path)
                target_dir = os.path.dirname(mask_path)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                torchvision.utils.save_image(mask, mask_path)

            if idx % 100 == 0 :
                print(f"batch {idx + 1}/{len(val_loader)}")

            if args.debug_mode : 
                break 

    print('Finished!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__) 
    # model & checkpoint
    parser.add_argument("--model_name", default="DETR_S_MultiHead_Register",)
    parser.add_argument("--checkpoint_file", default=None,)
    parser.add_argument("--class_num", type=int, default=1)

    # training schedule 
    parser.add_argument("--batch_size", type=int, default=40,)    
    parser.add_argument("--num_workers", type=int, default=24,)
    parser.add_argument("--half_precision", default=False, action="store_true",)


    # Data                        
    parser.add_argument("--data_dir", default='/mnt/nfs7/workshop/patch_level', help=("The directory containing dataset."))
    parser.add_argument("--output_dir", default='../temp')
    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument("--mag", type=int, default=100)

    parser.add_argument("--debug_mode", default=False, action="store_true",)

    args = parser.parse_args()

    args.mask_downsample = 8 if ('EfficientViT' in args.model_name) else 4
    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)


    print(args)

    inference(args)
