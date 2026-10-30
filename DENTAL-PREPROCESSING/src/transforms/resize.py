from torchvision.transforms import resize
from .zeropad import Zeropad 

def resize_transform(
    tensor, size  : tuple[int, int], padding_mode : str = "constant"

):
    _, img_height, img_width = tensor.shape 
    scale = min(size[0] / img_height, size [0] / img_width)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    reized_tensor = resize(tensor, [new_height, new_width])
    return Zeropad(reized_tensor,size [1], size[0], padding_mode = padding_mode)

class Resize:
    """ Resize a tensor image to a given size, with an option to keep dimension ratio. """

    def __init__(
            self,
            size: tuple[int, int],
            keep_ratio: bool = False,
            padding_mode: str = "constant",

    ):
        self.height, self.width = size 
        self.keep_aspect_ratio = keep_ratio
        self.padding_mode = padding_mode 


    def __call__(self, tensor):
        if self.keep_aspect_ratio:
            tensor = resize_transform(
                tensor, (self.height, self.width), self.padding_mode
            )
        else:
            tensor = resize(tensor, [self.height, self.width])
        return tensor
