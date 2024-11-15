"""
Defines a helper Model class to manage the setup, training, and testing of all models.
"""

from torch import Tensor, cat, no_grad
from torch.nn import (
    BatchNorm2d,
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
)
from torch.optim import Adam
from torchvision.transforms import ToTensor

from data import get_data_loaders


class UNetModel(Module):
    def __init__(self) -> None:
        super().__init__()

        self.max_pool = MaxPool2d(2)

        # downsampling convolutions
        self.down1 = Sequential(
            Conv2d(1, 4, 3, padding=1),
            BatchNorm2d(4),
            ReLU(),
            Conv2d(4, 4, 3, padding=1),
            BatchNorm2d(4),
            ReLU(),
        )

        self.down2 = Sequential(
            Conv2d(4, 8, 3, padding=1),
            BatchNorm2d(8),
            ReLU(),
            Conv2d(8, 8, 3, padding=1),
            BatchNorm2d(8),
            ReLU(),
        )

        self.down3 = Sequential(
            Conv2d(8, 16, 3, padding=1),
            BatchNorm2d(16),
            ReLU(),
            Conv2d(16, 16, 3, padding=1),
            BatchNorm2d(16),
            ReLU(),
        )

        self.down4 = Sequential(
            Conv2d(16, 32, 3, padding=1),
            BatchNorm2d(32),
            ReLU(),
            Conv2d(32, 32, 3, padding=1),
            BatchNorm2d(32),
            ReLU(),
        )

        # upsampling transposed convolutions
        self.up_transpose1 = ConvTranspose2d(
            32, 16, kernel_size=2, stride=2, output_padding=1
        )
        self.up1 = Sequential(
            Conv2d(32, 16, 3, padding=1),
            BatchNorm2d(16),
            ReLU(),
            Conv2d(16, 16, 3, padding=1),
            BatchNorm2d(16),
            ReLU(),
        )

        self.up_transpose2 = ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.up3 = Sequential(
            Conv2d(16, 8, 3, padding=1),
            BatchNorm2d(8),
            ReLU(),
            Conv2d(8, 8, 3, padding=1),
            BatchNorm2d(8),
            ReLU(),
        )

        self.up_transpose3 = ConvTranspose2d(8, 4, kernel_size=2, stride=2)
        self.up5 = Sequential(
            Conv2d(8, 4, 3, padding=1),
            BatchNorm2d(4),
            ReLU(),
            Conv2d(4, 4, 3, padding=1),
            BatchNorm2d(4),
            ReLU(),
        )

        # final layer with Sigmoid for normalized output
        self.up6 = Sequential(Conv2d(4, 1, 3, padding=1), Sigmoid())

    def forward(self, x) -> Tensor:
        # downsampling path
        x1 = self.down1(x)
        x2 = self.down2(self.max_pool(x1))
        x3 = self.down3(self.max_pool(x2))
        x4 = self.down4(self.max_pool(x3))

        # upsampling path with skip connections
        x5 = self.up_transpose1(x4)
        x5 = cat([x5, x3], dim=1)
        x5 = self.up1(x5)

        x6 = self.up_transpose2(x5)
        x6 = cat([x6, x2], dim=1)
        x6 = self.up3(x6)

        x7 = self.up_transpose3(x6)
        x7 = cat([x7, x1], dim=1)
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

    def __init__(
        self,
    ) -> None:
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
