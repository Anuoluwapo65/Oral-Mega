import torch  
import torch.utils.data as data  
from torch.utils.data import Dataset, DataLoader 
from pathlib import Path  
import random 
from typing import List 
from PIL import Image 
from datetime import datetime  
import numpy as np


class BaseCallback:
    """Base class for pipeline callbacks """

    def __init__(self, name:str = "BaseCallback"):
        self.name = name 

    def __call__(self, data, describe = "unknown-description"):
        """ call the callback and data description """
        raise NotImplementedError("subclasses must be implemnented this method")




class LogCallback(BaseCallback):
    """ Log callback for logging the data description and timestamp """

    def __init__(self, name:str = "Logstatsback"):
        super().__init__(name)

    def __call__(self, data, describe = "unkown description"):
        """ Log the data description and timestamp """
        array = np.array(data)
        stats_heads = (
            f"shape: {array.shape},"
            f"min: {array.min()},"
            f"max: {array.max()},"
            f"mean: {array.mean()}",
            f"std: {array.std()}",
            f"median: {np.median(array)}"
        )
        print(f"[{datetime.now()}: {describe}]-> {stats_heads}")

class SaveCallback(BaseCallback):
    """ Save callback for saving the data to a file """

    def __init__(self, save_dir:Path, name:str = "SaveCallback"):
        super().__init__(name)
        self.save_dir = save_dir 
        self.save_dir.mkdir(parents = True, exist_ok = True)

    def __call__(self, data, describe = "unknown - description"):
        """ convert the data to a numpy array and save it a file"""
        if hasattr(data, "cpu"):
            data = data.detach().cpu().numpy()



        if data.ndim == 2:
            data = np.expand_dims(data, axis = 0)

        elif data.ndim == 3:
            data = np.expand_dims(data, axis = 0)

        if not isinstance(data, np.ndarray):
            raise ValueError(f"Data must be a numpy array :{type(data)}")


        if data.dtype == np.float32 or data.dtype == np.float64:
            data = (data * 255).astype(np.uint8)

            data =(data - data.min()) / (data.max() - data.min()) * 255 
            data = data.astype(np.uint8)

        elif data.dtype == np.uint16:
            data = (data - data.min()) / ( data.max() - data.min()) * 65535
            data = data.astype(np.uint16)

        timestamp = datetime.now()
        filename = f"{describe.replace(' ', '_')}_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
        filepath = self.save_dir / filename
        np.save(filepath, data)




