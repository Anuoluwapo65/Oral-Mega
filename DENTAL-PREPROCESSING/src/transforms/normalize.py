import torch 
import numpy as np 

def minmax_normalize(image: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """ Normalize the image using min-max normalization techniques"""
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val + epsilon)

class MinMaxNormalize:
    """ Applies min-max normalization per sample """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon 

    def __call__(self, image: torch.Tensor)-> torch.Tensor:
        return minmax_normalize(image,self.epsilon)



    def zscore_normalize(image:torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """ Normalize the image using z-score normalization techniques """
        mean = image.mean(image.float())
        std = image.std(image.float())
        return (image - mean) / (std + epsilon)


class ZScoreNormalize:
    """ Applies z-score normalization per sample """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def __call__(self, image: torch.Tensor)-> torch.Tensor:
        return self.zscore_normalize(image, self.epsilon)

class ClipandScale:
    """ Clips the image to a given range and scale it to [0, 1]"""

    def __init__(
            self,
            min_percentile: float = 0.5,
            max_percentile : float = 99.5,
            epsilon: float = 1e-8
    ):

        self.min_percentile = min_percentile 
        self.max_percentile = max_percentile 
        self.epsilon = epsilon 


    def __call__(self, image: torch.Tensor)-> torch.Tensor:
        assert isinstance(image, torch.Tensor), "Input image must be a tensor"
        assert image.ndim == 3, "Input image must be a 3D tensor(C, H, W)"
        return self.clipandScale(image, self.min_percentile, self.max_percentile, self.epsilon)
