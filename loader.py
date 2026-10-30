from pathlib import Path 
import pydicom  
from PIL import Image 
from typing import Callable, List 
import numpy as np
from src.utils.register import register
from src.utils.register import REGISTRY




def load_file(file_path:Path):
    """ Load a file from using the appropriate registered loader."""
    return get_file_loader(file_path)(file_path)


def check_file_supported(file_path:Path):
    """ Check if the file is supported by any of the registered loaders."""
    extension = file_path.suffix.lower()
    return extension in REGISTRY["loaders"]

def get_file_loader(file_path:Path):
    """ Get the appropriate loader for a file."""
    extension = file_path.suffix.lower()
    return REGISTRY["loaders"][extension]

@register(".dcm", REGISTRY["loaders"])

def load_dicom(file_path:Path):
    """ Load a DICOM File and return a PIL Image."""
    dicom = pydicom.dcmread(file_path)
    return Image.fromarray(dicom.pixel_array)



@register(".png", REGISTRY["loaders"])
def load_png(file_path:Path):
    """ Load a PNG File and return a PIL Image."""
    img =  Image.open(file_path).convert("RGB")
    array = np.array(img, dtype = np.uint16)
    return Image.fromarray(array)