import torch 
import torch.nn.functional as F   
import torchvision 
from torchvision import transforms 
import os 
from pathlib import Path   
from copy import deepcopy   
from torch.utils.data import Dataset, DataLoader   
from loguru import logger 
from typing import List
from callbacks import BaseCallback
import random 
import callbacks 
from loader import load_file, check_file_supported 




class DentalDataset(Dataset):
    """Dental Dataset for dental Image preprocessing and segmentation task"""
    def __init__(self, data_dir, transform = None, callbacks: List[BaseCallback] = None):
        super(DentalDataset, self).__init__()
        """
        Args:
        data_dir (stringd): Directory with all the images and masks.
        transforms (callable, optional): Optional transform to be applied on a samples
        callbacks (List[BaseCallback], optional): List of all callbacks to be applied on the dataset"""
        self.data_dir = data_dir
        self.transform = transform
        self.callbacks = callbacks if callbacks is not None else []
        self.image_path = self._get_image_paths()

    def _get_image_paths(self):
        """ Get all the image paths from the data directory """
        datafiles = Path(self.data_dir)
        return [p for p in datafiles if p.is_file() and check_file_supported(p)]


    def __len__(self):
        """ Return the total number of samples in the dataset """
        return len(self.image_path)


    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image = load_file(image_path)


        for transform in self.transforms:
            for callback in self.callbacks:
                callback(
                    deepcopy(image),
                    describe=f"Applying {transform.__class__.__name__} on {image_path.stem}",
                    image_path=image_path
                )


            image = transform(image)


            for callback in self.callbacks:
                callback(
                    deepcopy(image),
                    describe = f"Applied {transform.__class__.__name__} on {image_path.stem}", 
                    image_path=image_path

                )

            return image 


