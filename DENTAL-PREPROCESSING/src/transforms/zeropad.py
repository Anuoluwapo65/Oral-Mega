import torch.nn.functional as F  

def zero_pad(tensor, width, height, padding_mode: str = "constant"):
    """ Zero pad a tensor to the given width and height """
    h, w = tensor.shape[-2:]
    target_h, target_w = height, width 
    pad_h = target_h - h
    pad_w = target_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top 
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left 

    if padding_mode == "constant":
        tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
    elif padding_mode == "reflect":
        tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    elif padding_mode == "replicate":
        tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

    return tensor
class Zeropad:
    """ A transform that zero pads a tensor to the given width and height"""

    def __init__(self, width, height, padding_mode: str = "constant"):
        self.width = width
        self.height = height
        self.padding_mode = padding_mode

    def __call__(self, tensor):
        return zero_pad(tensor, self.width, self.height, self.padding_mode)
