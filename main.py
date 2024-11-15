"""
Defines the experiments to be run.
"""

from datetime import datetime

from torch import cuda, no_grad
from torch.backends import mps
from torchsummary import summary

from data import append_to_file, save_image
from models import Autoencoder, UNet

if cuda.is_available():
    device = "cuda"
elif mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


@no_grad()
def qualitative_test(device, noise, autoencoder, unet):
    autoencoder.encoder.to(device)
    autoencoder.decoder.to(device)
    unet.model.to(device)

    autoencoder.encoder.eval()
    autoencoder.decoder.eval()
    unet.model.eval()

    i = 1
    for noisy_image, gt_image in autoencoder.test_loader:
        if i > 3:
            break

        noisy_image = noisy_image.to(device)
        gt_image = gt_image.to(device)

        encoded = autoencoder.encoder(noisy_image)
        decoded = autoencoder.decoder(encoded)

        unet_output = unet.model(noisy_image)

        pre = f"{noise} {i} "

        save_image(noisy_image[0], f"{pre} Noisy Image", "./images")
        
        save_image(
            gt_image[0],
            f"{pre} Ground Truth",
            "./images",
        )

        save_image(
            decoded[0],
            f"{pre} Autoencoder",
            "./images",
        )

        save_image(
            unet_output[0],
            f"{pre} UNet",
            "./images",
        )

        i += 1


filename = "results.txt"
append_to_file(filename, f"{datetime.now()}")

data_size = -1
noise_levels = [0.75, 0.85, 0.95]

for noise in noise_levels:
    autoencoder = Autoencoder()
    autoencoder.set_model()

    if noise == noise_levels[0]:
        print("Autoencoder Model Summary")
        summary(autoencoder.encoder, (1, 28, 28), device="cpu")
        summary(autoencoder.decoder, (1, 28, 28), device="cpu")

    autoencoder.set_data(noise=noise, size=data_size)
    autoencoder.train(device=device, epochs=20)

    autoencoder_loss = autoencoder.test(device=device)

    append_to_file(filename, f"autoencoder {noise} {autoencoder_loss}")

    ##########################################################################

    unet = UNet()

    unet.set_model()

    if noise == noise_levels[0]:
        print("UNet Model Summary")
        summary(unet.model, (1, 28, 28), device="cpu")

    unet.set_data(noise=noise, size=data_size)
    unet.my_train(device=device, epochs=20)

    unet_loss = unet.test(device=device)

    append_to_file(filename, f"unet {noise} {unet_loss}")

    ##########################################################################

    qualitative_test(device, noise, autoencoder, unet)
