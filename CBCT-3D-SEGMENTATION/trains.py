import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.utils.data as data 
from torch.utils.data import Dataset, DataLoader 
from pathlib import Path 
from loguru import logger 
import os 
import numpy as np 
import scipy.ndimage as ndimage 
from scipy.ndimage import zoom
import random
import json
from typing import List 


class DentalDataset(Dataset):

    def __init__(self, image_dir, mask_dir, data_dir, annotation_file, image_suffix = ".npy", transforms = None):
        super(DentalDataset, self).__init__()

        self.images, self.annotations = self.load_annotation(annotation_file)
        self.image_dir = image_dir 
        self.mask_dir = mask_dir 
        self.transforms = transforms 
        self.image_suffix = image_suffix


    def __len__(self):
        return len(self.images)



    def __getitem__(self, idx):
        """return image and mask for the given index"""
        return self.get_item(idx)



    def get_item(self, idx):
        # The annotation keys identify samples; no separate cache module is needed.
        key = self.images[idx]
        filename = key if Path(key).suffix else f"{key}{self.image_suffix}"
        image_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        # Load and return the image and mask
        return image_path, mask_path


    def load_annotation(self, annotation_file):
        with open(annotation_file, "r", encoding="utf-8") as f:
          data = json.load(f)
          keys = list(data.keys())
        return keys, data 


    def read_image(self, image_path):
        """Read the image from the given path and return the image as a numpy array."""
        return np.load(image_path)


    def transform(self, image):
        transformations = {"flip":self.flip, "rotate":self.rotate, "scale":self.scale, "normalize":self.normalize}

        if self.transforms:
            for transform in self.transforms:
                image = transformations[transform](image)
        return image


    def flip(self, image):

        return np.flip(image, axis = 0)

    def rotate(self, image):
        return np.rot90(image, k = 1, axes = (0, 1))

    def scale(self, image):
        scale_factor = random.uniform(0.8, 1.2)
        return zoom(image, scale_factor, order = 1)

    def normalize(self, image):
        min_val = np.min(image)
        max_val = np.max(image)
        return (image - min_val) / (max_val - min_val)




class RegionalDataset(DentalDataset):

    def __init__(self, image_dir, mask_dir, data_dir, annotation_file, image_suffix = ".npy", transforms = None, target_size = [100, 50, 50]):
        super(RegionalDataset, self).__init__(image_dir, mask_dir, data_dir, annotation_file, image_suffix, transforms)
        self.target_size = target_size


        self.boxes = self.load_boxes()

    def load_boxes(self):
        boxes = {}
        for key in self.annotations:
            boxes[key] = self.annotations[key]["boxes"]
        return boxes


    def get_item(self, idx):
        image_path, mask_path = super().get_item(idx)
        image = self.read_image(image_path)
        mask = self.read_image(mask_path)

        return image, mask




if __name__ == "__main__":
    # Example usage 
    image_dir = r"C:\path\to\images"
    mask_dir = r"C:\path\to\masks"
    data_dir = r"C:\path\to\data"
    annotation_file = r"C:\Users\ACE\ORAL-MEGA\data\annotations.json"

    dataset = RegionalDataset(image_dir, mask_dir, data_dir, annotation_file, transforms = ["flip", "normalize", "rotate", "scale"])
    
