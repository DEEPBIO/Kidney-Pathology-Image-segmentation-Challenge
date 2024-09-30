import torch.utils.data as data

from PIL import Image
import os
import os.path
import pdb
import math
import random

import cv2 as cv
import numpy as np
import torch
from scipy import ndimage
from skimage import transform as tf

class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotate(object):
    """
    Random Rotate 30, 45, 60
    """
    def __call__(self, img):
        if random.random() < 0.5:
            index = random.randint(0, 2)
            degree = [90, 180, 270]
            return img.rotate(degree[index])
        return img


class Rotate(object):
    """
    Rotate with given degrees
    """
    def __call__(self, img, degree):
        # index range = 0 ~ 12

        degree = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270]
        return img.rotate(degree[index])


class Shear(object):
    """
    Shear with given degrees
    """
    def __call__(self, img, index):
        # index = random.randint(0, 2)
        # index range = 0 ~ 6
        degree = [-25, -15, -5, 0, 5, 15, 25]
        radians = math.radians(degree[index])

        tf_matrix = tf.AffineTransform(shear=radians)
        modified = tf.warp(img, inverse_map=tf_matrix)
        modified = (modified * 255).astype(np.uint8)

        return Image.fromarray(modified)


class Flip(object):
    def __call__(self, img):
        return img.transpose(Image.FLIP_TOP_BOTTOM)


class RandomImageGradient(object):
    """
    Random Rotate 30, 45, 60
    """
    def __call__(self, img):
        if random.random() < 0.5:
            img_np = np.array(img)
            img_np_out = 0*np.uint8(img_np)
            for ch in range(3):
                img = img_np[:, :, ch]
                blur = cv.GaussianBlur(img, (11, 11), 0)

                sobelx64f = cv.Sobel(blur, cv.CV_64F, 1, 0, ksize=3)
                sobely64f = cv.Sobel(blur, cv.CV_64F, 0, 1, ksize=3)

                grad64f = np.sqrt(np.power(sobelx64f, 2)+np.power(sobely64f, 2))
                grad8u = np.uint8(grad64f)

                img_np_out[:, :, ch] = grad8u

            return Image.fromarray(img_np_out)
        return img


class CutOut(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int) : Number of patches to cut-out each image.
        length (int) : The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor) : Tensor image of size (C,H,W).
        Returns:
            Tensor : Image with n_holes of dimension length x length
                     cut out of it.
        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask


class RandomGaussianBlur(object):
    """
    Random Gaussian Blur
    """
    def __call__(self, img):

        if random.random() < 0.5:
            eps = 0.001
            sigma_sample = np.random.uniform(low=0, high=1)

            img_np = np.array(img)

            if sigma_sample > 0 + eps:
                for ch in range(3):
                    img_np[:, :, ch] = ndimage.gaussian_filter(img_np[:, :, ch],
                                                               sigma_sample)

            return Image.fromarray(img_np)

        return img


class RandomAdditiveGaussianNoise(object):
    """
    Random Additive Gaussian Noise
    """
    def __call__(self, img):

        if random.random() < 0.5:
            scale_std = 255 * 0.03

            img_np = np.array(img).astype(np.float64)
            img_noise = np.random.normal(0, scale_std, size=img_np.shape)

            img_np += img_noise
            img_np = np.clip(img_np, 0, 255)


            return Image.fromarray(img_np.astype(np.uint8))

        return img
