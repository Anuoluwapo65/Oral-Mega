import torch    
import numpy as np  

class AddChannelDim:
    """ A transform that adds a channel dimension to the input tensor"""
    def __call__(self, tensor):
        assert isinstance(tensor, torch.Tensor), "Input must be a tensor"

        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            return tensor.unsqueeze(0)
        else:
            raise ValueError(f"Input tensor must be 2D or 3D, but got {tensor.ndim}D")


class ToTensor:
    """ A transform that coverts a numpy array to a torch tensor"""
    def __call__(self, array, dtype = torch.float32):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).to(dtype)
        elif isinstance(array, torch.Tensor):
            return array.to(device = "CPU", dtype = torch.float32)
        else:
            raise ValueError(f" the input must be a numpy array or a tensor")
        
