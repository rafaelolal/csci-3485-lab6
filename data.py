"""
Manages the data for the experiments.
"""

from random import randint

from torch import Tensor, clamp, randn_like
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose


class NoisyMNISTDataset(Dataset):

    def __init__(
        self,
        train: bool,
        noise: float,
        transform: list,
        size: int,
        root: str = "./data/MNIST",
    ) -> None:

        self.dataset = MNIST(
            root=root, train=train, download=True, transform=Compose(transform)
        )
        self.noise = noise

        if size != -1:
            indices = [
                randint(0, len(self.dataset) - 1)
                for _ in range(min(size, len(self.dataset)))
            ]
            self.dataset = Subset(self.dataset, indices)

    def __getitem__(self, i) -> tuple[Tensor, Tensor]:
        image, label = self.dataset[i]

        # generate noise of image size scaled by self.noise
        noise = randn_like(image) * self.noise
        noisy_image = image + noise
        # maintain range from 0-1
        noisy_image = clamp(noisy_image, 0, 1)

        return noisy_image, image

    def __len__(self) -> int:
        return len(self.dataset)


def get_data_loaders(
    noise: float,
    size: int,
    transform: list = [],
) -> tuple[DataLoader]:
    test_dataset = NoisyMNISTDataset(
        noise=noise,
        transform=transform,
        size=size,
        train=False,
    )
    train_dataset = NoisyMNISTDataset(
        noise=noise,
        transform=transform,
        size=size,
        train=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader


def append_to_file(filename: str, data: any) -> None:
    """
    Appends data to a file.
    """

    with open(filename, "a") as file:
        file.write(str(data) + "\n")
