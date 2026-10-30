from .dataset import DentalDataset 
from .transforms import (
    ClipandScale,
    Resize,
    AddChannelDim,
    ToTensor,
)

from .generator import DataGenerator 
from .callbacks import SaveCallback, LogCallback 




def main():
    """ Main function to test the dataloader and transformations """
    dataset = DentalDataset(
        data_dir = "data/original",
        transforms = [
            ClipandScale(min_percentile = 0.5, max_percentile = 99.5),
            Resize(size = (256, 256)),
            AddChannelDim(),
            ToTensor()
        ]
    )

    generator = DataGenerator(
        dataset, batch_size = 4, shuffle = True, callbacks = [LogCallback()]

    )
    for batch in generator:
        print(f"Batch shape: {batch.shape}")

if __name__ == "__main__":
    main()
