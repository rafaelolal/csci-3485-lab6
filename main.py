"""
Defines the experiments to be run.
"""

from datetime import datetime

from torch.backends import mps
from torchsummary import summary

from data import append_to_file
from models import Autoencoder, UNet

if cuda.is_available():
    device = "cuda"
elif mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

filename = "results.txt"
append_to_file(filename, f"{datetime.now()}")

data_size = 100
noise_levels = [0.25, 0.5, 0.75]
for noise in noise_levels:
    autoencoder = Autoencoder()
    autoencoder.set_model(device=device)

    autoencoder.set_data(noise=noise, size=data_size)
    autoencoder.train(epochs=10)

    autoencoder_loss = autoencoder.test()

    append_to_file(filename, f"autoencoder {noise} {autoencoder_loss}")

    ##########################################################################

    unet = UNet()

    unet.set_model(device=device)
    unet.set_data(noise=noise, size=data_size)
    unet.my_train(epochs=10)

    unet_loss = unet.test()

    append_to_file(filename, f"unet {noise} {unet_loss}")
