import os
from random import randint

import matplotlib.pyplot as plt
from torch import Tensor, clamp, randn_like
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize
from torchvision.transforms.functional import center_crop

from torch import Tensor, cat, no_grad
from torch.nn import (
    Conv2d,
    ConvTranspose2d,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    MSELoss,
    ReLU,
    Sequential,
    Sigmoid,
    BatchNorm2d
)
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import ToTensor, Resize

from datetime import datetime

from torch import cuda, no_grad
from torch.backends import mps
from torchsummary import summary

if cuda.is_available():
    device = "cuda"
elif mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


# data.py

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
    # Ensure noise level is within the valid range
    assert 0 <= noise <= 1, "Noise level (mu) must be between 0 and 1."

    # Ensure proper dimensions
    transform = [Resize((28, 28))] + transform

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
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Reshape and convert to numpy for plotting
    image = image.reshape(28, 28).detach().cpu().numpy()

    # Create the plot
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    plt.title(f"{title}")
    plt.axis("off")

    # Save the image
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(folder, filename))
    plt.close()



# models.py

class UNetModel(Module):
    def __init__(self) -> None:
        super().__init__()

        self.max_pool = MaxPool2d(2)

        # Downsampling convolutions
        self.down1 = Sequential(Conv2d(1, 4, 3, padding=1), BatchNorm2d(4), ReLU())
        self.down2 = Sequential(Conv2d(4, 4, 3, padding=1), BatchNorm2d(4), ReLU())

        self.down3 = Sequential(Conv2d(4, 8, 3, padding=1), BatchNorm2d(8), ReLU())
        self.down4 = Sequential(Conv2d(8, 8, 3, padding=1), BatchNorm2d(8), ReLU())

        self.down5 = Sequential(Conv2d(8, 16, 3, padding=1), BatchNorm2d(16), ReLU())
        self.down6 = Sequential(Conv2d(16, 16, 3, padding=1), BatchNorm2d(16), ReLU())

        self.down7 = Sequential(Conv2d(16, 32, 3, padding=1), BatchNorm2d(32), ReLU())
        self.down8 = Sequential(Conv2d(32, 32, 3, padding=1), BatchNorm2d(32), ReLU())

        # Upsampling transposed convolutions and convolutions
        self.up_transpose1 = ConvTranspose2d(32, 16, kernel_size=2, stride=2, output_padding=1)
        self.up1 = Sequential(
            Conv2d(32, 16, 3, padding=1), BatchNorm2d(16), ReLU(),
            Conv2d(16, 16, 3, padding=1), BatchNorm2d(16), ReLU()  # Added extra Conv2d
        )

        self.up_transpose2 = ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.up3 = Sequential(
            Conv2d(16, 8, 3, padding=1), BatchNorm2d(8), ReLU(),
            Conv2d(8, 8, 3, padding=1), BatchNorm2d(8), ReLU()  # Added extra Conv2d
        )

        self.up_transpose3 = ConvTranspose2d(8, 4, kernel_size=2, stride=2)
        self.up5 = Sequential(
            Conv2d(8, 4, 3, padding=1), BatchNorm2d(4), ReLU(),
            Conv2d(4, 4, 3, padding=1), BatchNorm2d(4), ReLU()  # Added extra Conv2d
        )
        self.up6 = Sequential(Conv2d(4, 1, 3, padding=1), Sigmoid())  # Final layer with Sigmoid for normalized output

    def forward(self, x) -> Tensor:
        # Downsampling path
        x1 = self.down1(x)
        x1 = self.down2(x1)

        x2 = self.max_pool(x1)
        x2 = self.down3(x2)
        x2 = self.down4(x2)

        x3 = self.max_pool(x2)
        x3 = self.down5(x3)
        x3 = self.down6(x3)

        x4 = self.max_pool(x3)
        x4 = self.down7(x4)
        x4 = self.down8(x4)

        # Upsampling path with skip connections
        x5 = self.up_transpose1(x4)
        x3_cropped = center_crop(x3, x5.shape[2:])
        x5 = cat([x5, x3_cropped], dim=1)
        x5 = self.up1(x5)

        x6 = self.up_transpose2(x5)
        x2_cropped = center_crop(x2, x6.shape[2:])
        x6 = cat([x6, x2_cropped], dim=1)
        x6 = self.up3(x6)

        x7 = self.up_transpose3(x6)
        x1_cropped = center_crop(x1, x7.shape[2:])
        x7 = cat([x7, x1_cropped], dim=1)
        x7 = self.up5(x7)
        x7 = self.up6(x7)

        return x7



class UNet:
    def set_model(self) -> None:
        self.model = UNetModel()

    def set_data(self, noise: float, size: int) -> None:
        self.train_loader, self.test_loader = get_data_loaders(
            noise=noise,
            size=size,
            transform=[ToTensor()],
        )

    def my_train(self, device: str, epochs: int) -> None:
        self.model.to(device)
        self.model.train()

        optimizer = Adam(self.model.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        loss_function = MSELoss()

        for _ in range(epochs):
            for noise_image, image in self.train_loader:
                noise_image = noise_image.to(device)
                image = image.to(device)

                output = self.model(noise_image)
                loss = loss_function(output, image)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

    @no_grad()
    def test(self, device: str) -> float:
        self.model.to(device)
        self.model.eval()

        loss = 0
        loss_function = MSELoss()

        for noisy_image, gt_image in self.test_loader:
            noisy_image = noisy_image.to(device)
            gt_image = gt_image.to(device)

            output = self.model(noisy_image)

            loss += loss_function(output, gt_image)

        return loss / len(self.test_loader)


class Autoencoder:

    def __init__(self) -> None:
        pass

    def set_model(self) -> None:
        self.encoder = Sequential(
            Flatten(),
            Linear(784, 112),
            ReLU(),
            Linear(112, 56),
            ReLU(),
            Linear(56, 28),
            ReLU(),
        )

        self.decoder = Sequential(
            Linear(28, 56),
            ReLU(),
            Linear(56, 112),
            ReLU(),
            Linear(112, 784),
            Sigmoid(),
        )

    def set_data(self, noise: float, size: int) -> None:
        self.train_loader, self.test_loader = get_data_loaders(
            noise=noise,
            transform=[ToTensor()],
            size=size,
        )

    def train(self, device: str, epochs: int) -> None:
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.train()
        self.decoder.train()

        optimizer = Adam(
            [*self.encoder.parameters(), *self.decoder.parameters()], lr=1e-3
        )
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        loss_function = MSELoss()

        for _ in range(epochs):
            for noisy_image, gt_image in self.train_loader:
                noisy_image = noisy_image.to(device)
                gt_image = gt_image.to(device)
                gt_image = gt_image.view(-1, 28 * 28)

                latent = self.encoder(noisy_image)
                output = self.decoder(latent)

                loss = loss_function(output, gt_image)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

    @no_grad()
    def test(self, device: str) -> float:
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()

        loss = 0
        loss_function = MSELoss()

        for noise_image, gt_image in self.test_loader:
            noise_image = noise_image.to(device)
            gt_image = gt_image.to(device)
            gt_image = gt_image.view(-1, 28 * 28)

            latent = self.encoder(noise_image)
            output = self.decoder(latent)

            loss += loss_function(output, gt_image)

        return loss / len(self.test_loader)
    
# main.py

# Define the directory path
image_dir = '/content/images/'

# Create the directory if it doesn't exist
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

filename = "results.txt"
append_to_file(filename, f"{datetime.now()}")

data_size = -1
noise_levels = [0.25, 0.6, 0.9]
for noise in noise_levels:

    autoencoder = Autoencoder()
    autoencoder.set_model()
    autoencoder.encoder.to(device)
    autoencoder.decoder.to(device)

    if noise == noise_levels[0]:
        print("Autoencoder Model Summary")
        print(f"Running on: {device}")
        summary(autoencoder.encoder, (1, 28, 28), device=device)
        summary(autoencoder.decoder, (1, 28, 28), device=device)

    autoencoder.set_data(noise=noise, size=data_size)
    autoencoder.train(device=device, epochs=20)

    autoencoder_loss = autoencoder.test(device=device)
    print(f"{noise}: Autoencoder loss: {autoencoder_loss}")

    append_to_file(filename, f"autoencoder {noise} {autoencoder_loss}")

    ##########################################################################

    unet = UNet()

    unet.set_model()
    unet.model.to(device)
    print(f"U-Net running on: {device}")

    if noise == noise_levels[0]:
        print("UNet Model Summary")
        summary(unet.model, (1, 28, 28), device=device)

    unet.set_data(noise=noise, size=data_size)
    unet.my_train(device=device, epochs=20)

    unet_loss = unet.test(device=device)
    print(f"{noise}: U-net loss: {unet_loss}")

    append_to_file(filename, f"unet {noise} {unet_loss}")

    ##########################################################################

    # autoencoder.encoder.to(device)
    # autoencoder.decoder.to(device)
    # unet.model.to(device)

    autoencoder.encoder.eval()
    autoencoder.decoder.eval()
    unet.model.eval()

    i = 1
    with no_grad():
        for noisy_image, gt_image in autoencoder.test_loader:
            if i > 3:
                break

            noisy_image = noisy_image.to(device)
            gt_image = gt_image.to(device)

            encoded = autoencoder.encoder(noisy_image)
            decoded = autoencoder.decoder(encoded)
            decoded = decoded[0]

            unet_output = unet.model(noisy_image)
            unet_output = unet_output[0]

            gt_image = gt_image[0]

            pre = f"{noise} {i} "
            save_image(
                gt_image,
                f"{pre} Ground Truth",
                "/content/images",
            )

            save_image(
                decoded,
                f"{pre} Autoencoder",
                "/content/images",
            )

            save_image(
                unet_output,
                f"{pre} UNet",
                "/content/images",
            )

            i += 1