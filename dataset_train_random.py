#!/usr/bin/env python3
"""Define custom Dataset class.
"""
import os
import random
import pdb
import PIL
from PIL import Image
import torch
import torch.utils.data
import torchvision
import numpy as np

import my_folder

DATA_SIZE = 2048
DATA_MAG = 200
#MASK_DOWNSAMPLE = 4
#self.patch_size_in_image = 1024
#IMAGE_DOWNSAMPLE = 2

class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                    PatchDir='/mnt/nfs7/workshop/kidney_KPI_2024/patch_level',
                    patch_size=512,
                    mag=100,
                    mask_downsample=4, # 4 in general, 8 for EfficientViT
                    Train = True,
                    PatchTransform=None,
                    RotateFlipTransform=True,
                    RandomCrop=True,
                    RandomMpp=True,
                    CenterCrop=False,
                    debug = False,
                    imagenet_dir = '/mnt/nfs4/users/old_data/90_user/users/iypaik/ImageNet21K_512',
                    imagenet_prob = 0., ):

        if Train : 
            PatchDir = os.path.join(PatchDir, 'trn_data')
        else : 
            PatchDir = os.path.join(PatchDir, 'val_data')
            imagenet_prob = 0.
        
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
        self._imagenet_dir = imagenet_dir
        self._imagenet_cla_list = os.listdir(imagenet_dir) if imagenet_dir is not None else None 
        self._imagenet_prob = imagenet_prob 
    
        # /mnt/nfs7/workshop/patch_level/trn_data/56Nx/12_116/img/56Nx_12_116_4_4096_0_img.jpg
        # /mnt/nfs7/workshop/patch_level/trn_data/56Nx/12_116/mask/56Nx_12_116_4_4096_0_mask.jpg
        for i, class_ in enumerate(os.listdir(PatchDir)):
            class_dir = os.path.join(PatchDir, class_)
            for slide_ in os.listdir(class_dir):
                slide_dir = os.path.join(class_dir, slide_)
                img_dir = os.path.join(slide_dir, 'img')
                mask_dir = os.path.join(slide_dir, 'mask')
                file_list = os.listdir(img_dir)
                for file_ in file_list:
                    img_path = os.path.join(img_dir, file_)
                    mask_path = os.path.join(mask_dir, file_.replace('img','mask'))
                    if not os.path.isfile(mask_path) : 
                        print(f'{mask_path} not exists.')
                        pdb.set_trace()
                    self._data_list.append(img_path)
                    self._target_list.append(mask_path)
            
                if len(self) >= 100 and debug : 
                    break 

        
        print(f'Number of data: {len(self)}')
        if Train: 
            print('Hardcoding: use both train and validation dataset')
            for i, class_ in enumerate(os.listdir(PatchDir.replace('trn_data', 'val_data'))):
                class_dir = os.path.join(PatchDir.replace('trn_data', 'val_data'), class_)
                for slide_ in os.listdir(class_dir):
                    slide_dir = os.path.join(class_dir, slide_)
                    img_dir = os.path.join(slide_dir, 'img')
                    mask_dir = os.path.join(slide_dir, 'mask')
                    file_list = os.listdir(img_dir)
                    for file_ in file_list:
                        img_path = os.path.join(img_dir, file_)
                        mask_path = os.path.join(mask_dir, file_.replace('img','mask'))
                        if not os.path.isfile(mask_path) : 
                            print(f'{mask_path} not exists.')
                            pdb.set_trace()
                        self._data_list.append(img_path)
                        self._target_list.append(mask_path)
            print(f'Number of data (2): {len(self)}')

    def __len__(self):
        """Return the number of the patches to be .
        """
        return len(self._data_list) if self._train else len(self._data_list)*(DATA_SIZE//self.patch_size_in_image)**2

    def __getitem__(self, index):
        if random.random() < self._imagenet_prob: 
            cla = random.choice(self._imagenet_cla_list)
            subdir = os.path.join(self._imagenet_dir, cla)
            image_name = random.choice(os.listdir(subdir))
            image_path = os.path.join(subdir, image_name)

            patch = PIL.Image.open(image_path).convert('RGB')
            patch = patch.resize((self.patch_size, self.patch_size), PIL.Image.BILINEAR)
            if self._patchtrans:
                patch = self._patchtrans(patch)
            patch = self._totensor(patch)
            mask = torch.zeros(1, self.patch_size//self.mask_downsample, self.patch_size//self.mask_downsample)
            return patch, mask 

        if self._train: 
            patch_path = self._data_list[index]
            target_path = self._target_list[index]

            patch = PIL.Image.open(patch_path).convert('RGB')
            mask = PIL.Image.open(target_path)
        
        else : 
            N = len(self._data_list)
            position, data_index = index//N, index%N
            # 데이터 개수 1000개 
            # len(dataset) = 4000
            # index = 0~3999
            # position = 0~3
            # data_index = 0~999
            # position_x = 0~1
            # position_y = 0~1 
            position_x, position_y = position//(DATA_SIZE//self.patch_size_in_image), position%(DATA_SIZE//self.patch_size_in_image)
            x_start, y_start = position_x*self.patch_size_in_image, position_y*self.patch_size_in_image
            x_end, y_end = x_start+self.patch_size_in_image, y_start+self.patch_size_in_image

            patch_path = self._data_list[data_index]
            target_path = self._target_list[data_index]
            patch = PIL.Image.open(patch_path).convert('RGB')
            mask = PIL.Image.open(target_path)

        if self._train: 
            # If rotate, rotate first before crop
            if self._rotfliptrans:
                if random.random() < 0.5:
                    patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
                    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                    '''patch = my_folder.Flip()(patch)
                    mask = my_folder.Flip()(mask)'''

                # Random Rotate
                degree = random.randint(0,359)
                patch = patch.rotate(degree)
                mask = mask.rotate(degree)
                '''index = random.randint(0, 12)
                patch = my_folder.Rotate()(patch, index)
                mask = my_folder.Rotate()(mask, index)'''

            shake_size = int(self.patch_size_in_image * 0.2) # 20 percent rule
            index_size = max(DATA_SIZE - self.patch_size_in_image - shake_size, 0)
                
            if self._randomcrop:
                if self._randommpp:
                    index_x = random.randint(0, index_size)
                    index_y = random.randint(0, index_size)
                    
                    if random.random() < 0.5:
                        random_pixel = np.random.uniform(-shake_size, shake_size, 2).astype(np.int8)
                        x_delta, y_delta = random_pixel

                    else:
                        x_delta, y_delta = 0, 0
                    
                else:
                    index_x = random.randint(0, DATA_SIZE - self.patch_size_in_image)
                    index_y = random.randint(0, DATA_SIZE - self.patch_size_in_image)

            elif self._centercrop:
                index_x = (DATA_SIZE - self.patch_size_in_image) // 2
                index_y = index_x

                x_delta, y_delta = 0, 0

            else:
                raise ValueError('plz check your condition, randomcrop or centercrop should be included')

            index_x_ = index_x # // 2
            index_y_ = index_y # // 2

            patch = patch.crop((index_x,
                                index_y,
                                index_x + self.patch_size_in_image + x_delta,
                                index_y + self.patch_size_in_image + y_delta)) 
            patch = patch.resize((self.patch_size_in_image, self.patch_size_in_image), PIL.Image.BILINEAR)
            mask = mask.crop((index_x_,
                                index_y_,
                                index_x_ + ((self.patch_size_in_image + x_delta)), #  // 2
                                index_y_ + ((self.patch_size_in_image + y_delta) )))  # // 2
        else: 
            patch = patch.crop((x_start, y_start, x_end, y_end))
            mask = mask.crop((x_start, y_start, x_end, y_end))


        patch = patch.resize((self.patch_size, self.patch_size), PIL.Image.BILINEAR) 
        mask = mask.resize((self.patch_size // self.mask_downsample, self.patch_size // self.mask_downsample), PIL.Image.NEAREST) 

        if self._patchtrans:
            patch = self._patchtrans(patch)

        patch = self._totensor(patch)
        mask = self._totensor(mask)

        return patch, mask


class Dataset_mpp(torch.utils.data.Dataset):

    def __init__(self,
                    PatchDir='/mnt/nfs7/workshop/kidney_KPI_2024/patch_level',
                    patch_size=512,
                    mag=100,
                    mask_downsample=4, # 4 in general, 8 for EfficientViT
                    Train = True,
                    PatchTransform=None,
                    RotateFlipTransform=True,
                    RandomCrop=True,
                    RandomMpp=True,
                    CenterCrop=False,
                    debug = False,
                    imagenet_dir = '/mnt/nfs4/users/old_data/90_user/users/iypaik/ImageNet21K_512',
                    imagenet_prob = 0., ):

        if Train : 
            PatchDir = os.path.join(PatchDir, 'trn_data')
        else : 
            PatchDir = os.path.join(PatchDir, 'val_data')
            imagenet_prob = 0.
        

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
        self._NEP_list = [] 
        self._imagenet_dir = imagenet_dir
        self._imagenet_cla_list = os.listdir(imagenet_dir) if imagenet_dir is not None else None 
        self._imagenet_prob = imagenet_prob 
    
        # /mnt/nfs7/workshop/patch_level/trn_data/56Nx/12_116/img/56Nx_12_116_4_4096_0_img.jpg
        # /mnt/nfs7/workshop/patch_level/trn_data/56Nx/12_116/mask/56Nx_12_116_4_4096_0_mask.jpg
        for i, class_ in enumerate(os.listdir(PatchDir)):
            is_NEP = (class_ == 'NEP25')
            class_dir = os.path.join(PatchDir, class_)
            for slide_ in os.listdir(class_dir):
                slide_dir = os.path.join(class_dir, slide_)
                img_dir = os.path.join(slide_dir, 'img')
                mask_dir = os.path.join(slide_dir, 'mask')
                file_list = os.listdir(img_dir)
                for file_ in file_list:
                    img_path = os.path.join(img_dir, file_)
                    mask_path = os.path.join(mask_dir, file_.replace('img','mask'))
                    if not os.path.isfile(mask_path) : 
                        print(f'{mask_path} not exists.')
                        pdb.set_trace()
                    self._data_list.append(img_path)
                    self._target_list.append(mask_path)
                    self._NEP_list.append(is_NEP)
                if len(self) >= 100 and debug : 
                    break 

        
        print(f'Number of data: {len(self)}')
        '''if Train: 
            print('Hardcoding: use both train and validation dataset')
            for i, class_ in enumerate(os.listdir(PatchDir.replace('trn_data', 'val_data'))):
                class_dir = os.path.join(PatchDir.replace('trn_data', 'val_data'), class_)
                for slide_ in os.listdir(class_dir):
                    slide_dir = os.path.join(class_dir, slide_)
                    img_dir = os.path.join(slide_dir, 'img')
                    mask_dir = os.path.join(slide_dir, 'mask')
                    file_list = os.listdir(img_dir)
                    for file_ in file_list:
                        img_path = os.path.join(img_dir, file_)
                        mask_path = os.path.join(mask_dir, file_.replace('img','mask'))
                        if not os.path.isfile(mask_path) : 
                            print(f'{mask_path} not exists.')
                            pdb.set_trace()
                        self._data_list.append(img_path)
                        self._target_list.append(mask_path)
            print(f'Number of data (2): {len(self)}')'''

    def __len__(self):
        """Return the number of the patches to be .
        """
        return len(self._data_list) #if self._train else len(self._data_list)*(DATA_SIZE//self.patch_size_in_image)**2

    def __getitem__(self, index):
        is_NEP = self._NEP_list[index]

        if random.random() < self._imagenet_prob: 
            cla = random.choice(self._imagenet_cla_list)
            subdir = os.path.join(self._imagenet_dir, cla)
            image_name = random.choice(os.listdir(subdir))
            image_path = os.path.join(subdir, image_name)

            patch = PIL.Image.open(image_path).convert('RGB')
            patch = patch.resize((1024, 1024), PIL.Image.BILINEAR)
            if self._patchtrans:
                patch = self._patchtrans(patch)
            patch = self._totensor(patch)
            mask = torch.zeros(1, 1024, 1024)
            return patch, mask, torch.BoolTensor([False])

        patch_path = self._data_list[index]
        target_path = self._target_list[index]
        patch = PIL.Image.open(patch_path).convert('RGB')
        mask = PIL.Image.open(target_path)

        if self._train: 
            # If rotate, rotate first before crop
            if self._rotfliptrans:
                if random.random() < 0.5:
                    patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
                    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                    '''patch = my_folder.Flip()(patch)
                    mask = my_folder.Flip()(mask)'''

                # Random Rotate
                degree = random.randint(0,359)
                patch = patch.rotate(degree)
                mask = mask.rotate(degree)
                '''index = random.randint(0, 12)
                patch = my_folder.Rotate()(patch, index)
                mask = my_folder.Rotate()(mask, index)'''

            shake_size = int(self.patch_size_in_image * 0.2) # 20 percent rule
            index_size = max(DATA_SIZE - self.patch_size_in_image - shake_size, 0)
                
            if self._randomcrop:
                if self._randommpp:
                    index_x = random.randint(0, index_size)
                    index_y = random.randint(0, index_size)
                    
                    if random.random() < 0.5:
                        random_pixel = np.random.uniform(-shake_size, shake_size, 2).astype(np.int8)
                        x_delta, y_delta = random_pixel

                    else:
                        x_delta, y_delta = 0, 0
                    
                else:
                    index_x = random.randint(0, DATA_SIZE - self.patch_size_in_image)
                    index_y = random.randint(0, DATA_SIZE - self.patch_size_in_image)

            elif self._centercrop:
                index_x = (DATA_SIZE - self.patch_size_in_image) // 2
                index_y = index_x

                x_delta, y_delta = 0, 0

            else:
                raise ValueError('plz check your condition, randomcrop or centercrop should be included')

            index_x_ = index_x # // 2
            index_y_ = index_y # // 2

            patch = patch.crop((index_x,
                                index_y,
                                index_x + self.patch_size_in_image + x_delta,
                                index_y + self.patch_size_in_image + y_delta)) 
            patch = patch.resize((self.patch_size_in_image, self.patch_size_in_image), PIL.Image.BILINEAR)
            mask = mask.crop((index_x_,
                                index_y_,
                                index_x_ + ((self.patch_size_in_image + x_delta)), #  // 2
                                index_y_ + ((self.patch_size_in_image + y_delta) )))  # // 2

        mpp_patch_size = 1104 if is_NEP else self.patch_size # round(self.patch_size*0.11975/0.1109)
        patch = patch.resize((mpp_patch_size, mpp_patch_size), PIL.Image.BILINEAR) 
        if self._train: 
            mask = mask.resize((mpp_patch_size, mpp_patch_size), PIL.Image.NEAREST) 
            patch = torchvision.transforms.functional.center_crop(patch, self.patch_size)
            mask = torchvision.transforms.functional.center_crop(mask, self.patch_size)
        else: 
            pad_size = 1104 - mpp_patch_size
            patch = torchvision.transforms.functional.pad(patch, (0, 0, pad_size,pad_size)) # (left, top, right, bottom)
            mask = mask.resize((self.patch_size, self.patch_size), PIL.Image.NEAREST) 

        if self._patchtrans:
            patch = self._patchtrans(patch)

        patch = self._totensor(patch)
        mask = self._totensor(mask)

        return patch, mask, torch.BoolTensor([is_NEP])
    

if __name__ == '__main__' : 
    print(55)
    from torchvision.utils import save_image
    import pdb 
    dataset = Dataset(Train=False, imagenet_prob = 0, mag=100, patch_size=1024)

    pdb.set_trace()
    import os
    import shutil
    import random
    os.mkdir('./patch_sample')
    os.mkdir('./mask_sample')
    for _ in range(10): 
        i = random.randint(0, len(dataset)-1)
        shutil.copy(dataset._data_list[i], './patch_sample')
        shutil.copy(dataset._target_list[i], './mask_sample')

    #for patch,mask in dataset: pdb.set_trace() if (mask==0).float().mean()+(mask==1).float().mean() < 0.99 else print((mask==0).float().mean()+(mask==1).float().mean())
    '''
    patch, mask, is_NEP = dataset[1]
    save_image(patch, 'test_img.png')
    save_image(mask, 'test_mask.png')
    print(patch.shape)
    print(mask.shape)
    print(is_NEP)
    pdb.set_trace()'''

