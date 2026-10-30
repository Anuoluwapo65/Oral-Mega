import torch 
import random 
import torch.utils.data as data 
from torch.utils.data import Dataset 


def dental_data_generator(dataset: Dataset, batch_size : int, shuffle: bool = True):
    """ A generator function for generating the dataset
    Args:
    dataset: An object that implement __len__ and __getitem__.
    batch_size: number of sampls per batch
    shuffle: whether to shuffle the data at at the start of each epoch"""

    num_samples = len(dataset)
    indices = list(range(num_samples))

    if shuffle:
        random.shuffle(indices)

    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        batch = [dataset[idx] for idx in batch_indices]

        if batch:
            yield torch.stack(batch)
            