"""
Manages the data for the experiments.
"""

from os import path
from random import randint

import matplotlib.pyplot as plt
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

        # Generate noise with the same shape as the image
        noise = randn_like(image)

        # Apply noise using the formula: Y = (1 - mu) * I + mu * N
        noisy_image = (1 - self.noise) * image + self.noise * noise
        noisy_image = clamp(noisy_image, 0, 1)

        return noisy_image, image

    def __len__(self) -> int:
        return len(self.dataset)


def get_data_loaders(
    noise: float,
    size: int,
    transform: list = [],
) -> tuple[DataLoader, DataLoader]:

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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    return train_loader, test_loader


def append_to_file(filename: str, data: any) -> None:
    """
    Appends data to a file.
    """

    with open(filename, "a") as file:
        file.write(str(data) + "\n")


def save_image(image, title, folder):
    # reshape and convert to numpy for plotting
    image = image.reshape(28, 28).detach().cpu().numpy()

    # create the plot
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    plt.title(f"{title}")
    plt.axis("off")

    # save the image
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(path.join(folder, filename))
    plt.close()
