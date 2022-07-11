"""Module builds, trains and stores a DCGAN based model on MNIST data."""

from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os

from torch import nn
from torch import cuda
from torch.optim import Adam
from tqdm import tqdm
from torch import FloatTensor
from torch.autograd import Variable
from numpy import random
import cv2
from time import strftime
from torchvision.utils import save_image
import torch
from torch.utils.tensorboard import SummaryWriter

import logging
logger = logging.getLogger(name="DCGAN")
os.makedirs(".logs/dcgan/", exist_ok=True)
log_file = f".logs/dcgan/training_.log"
logging.basicConfig(
    level=logging.INFO,
    filename=log_file,
    filemode='w',
    format="%(name)s - %(levelname)s - %(message)s",
    )


class Generator(nn.Module):

    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        img_size, channels = img_shape[1], img_shape[0]
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x
    
class Discriminator(nn.Module):

    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        img_size, channels = img_shape[1], img_shape[0]

        def discriminator_block(in_filters, out_filters, normalize=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if normalize:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, normalize=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = img_size // 2 ** 4
        self.output_layer = nn.Sequential(
            nn.Linear(128 * ds_size **2, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.shape[0], -1)
        x = self.output_layer(x)

        return x

def get_data(root, img_size, batch_size):
    """Returns MNIST dataset as a dataloader"""

    os.makedirs(root, exist_ok=True)
    train_data = DataLoader(
        datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])]
                ),
        ),
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_data

def get_loss():
    return nn.BCELoss()

def get_device():
    return "cuda" if cuda.is_available() else "cpu"

def get_optimizer(model, lr, b1, b2):
    return Adam(model.parameters(), lr=lr, betas=(b1, b2))

def generate_noise(noise_shape, Tensor):
    return Variable(Tensor(random.normal(0, 1, noise_shape)))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def main(latent_dim, img_shape, batch_size, lr, b1, b2, root, n_epochs, sampling_interval, img_dir):
    writer = SummaryWriter(log_dir=f".logs/dcgan/{strftime('%d%m%Y_%H%M%S')}")

    img_size, channels = img_shape[1], img_shape[0]
    logger.info(f"Image Size: {img_size} | Image Channels: {channels}")

    data_loader = get_data(root, img_size, batch_size)
    logger.info(f"Training data initialized with batch size of {batch_size}")

    device = get_device()
    logger.info(f"Available device for training: {device}")

    example_input, _ = iter(data_loader).next()

    generator = Generator(latent_dim, img_shape)
    p_count = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    logger.info(f"Generator Model Initialized, total trainable parameters: {p_count}")
    writer.add_graph(generator, generate_noise((example_input.shape[0], latent_dim), FloatTensor))

    discriminator = Discriminator(img_shape)
    p_count = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    logger.info(f"Discriminator Model Initialized, total trainable parameters: {p_count}")
    writer.add_graph(discriminator, example_input)

    loss = get_loss()

    if device == "cuda":
        generator.cuda()
        discriminator.cuda()
        loss.cuda()

    g_optim = get_optimizer(generator, lr, b1, b2)
    d_optim = get_optimizer(discriminator, lr, b1, b2)
    logger.info(f"Training optimized initialized with Learning rate: {lr}")

    Tensor = cuda.FloatTensor if device == "cuda" else FloatTensor
    os.makedirs(img_dir, exist_ok=True)

    batch_size = len(data_loader)
    curr_min_g_loss = float('inf')
    curr_min_d_loss = float('inf')
    os.makedirs(".model/dcgan/", exist_ok=True)

    for epoch in tqdm(range(n_epochs)):
        mean_g_loss, mean_d_loss = 0, 0
        logger.info(f"Epoch: {epoch}")
        for batch, (imgs, _) in enumerate(tqdm(data_loader)):

            # Define labels
            real = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Setup input
            real_imgs = Variable(imgs.type(Tensor))

            # Train Generator
            g_optim.zero_grad()

            # Generate noise
            z = generate_noise((imgs.shape[0], latent_dim), Tensor)

            gen_imgs = generator(z)

            g_loss = loss(discriminator(gen_imgs), real)

            g_loss.backward()
            g_optim.step()

            mean_g_loss += g_loss.item()
            if g_loss.item() < curr_min_g_loss:
                curr_min_g_loss = g_loss.item()
                torch.save(generator, ".model/dcgan/generator.pt")
            
            # Train Discriminator
            d_optim.zero_grad()

            real_loss = loss(discriminator(real_imgs), real)
            fake_loss = loss(discriminator(gen_imgs.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            d_optim.step()

            mean_d_loss += d_loss.item()
            if d_loss.item() < curr_min_d_loss:
                curr_min_d_loss = d_loss.item()
                torch.save(discriminator, ".model/dcgan/discriminator.pt")

            curr_iter = (epoch * batch_size) + batch

            if curr_iter % sampling_interval == 0:
                writer.add_scalar("Generator Loss", g_loss.item(), curr_iter)
                writer.add_scalar("Discriminator Loss", d_loss.item(), curr_iter)
                logger.info(f"Iteration: {curr_iter}/{n_epochs*batch_size} | G-Loss: {g_loss.item()} | D-Loss: {d_loss.item()}")
                save_image(gen_imgs.data[:25], os.path.join(img_dir, f"{curr_iter}.png"), nrow=5, normalize=True)
                img_to_show = cv2.imread(os.path.join(img_dir, f"{curr_iter}.png"), 1)
                img_to_show = cv2.resize(img_to_show, (5*5*img_size, 5*5*img_size))
                cv2.imshow("DCGAN Generator Samples", img_to_show)
                cv2.waitKey(10)

        mean_g_loss, mean_d_loss = mean_g_loss/batch_size, mean_d_loss/batch_size
        logger.info(f"\nMean Generator Loss: {mean_g_loss} \nMean Discriminator: {mean_d_loss}")
        
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logger.info("Starting Model Training")
    main(
        root=".data/mnist/",
        img_dir = ".data/output/images/dcgan/mnist",
        latent_dim=100,
        img_shape=(1, 32, 32),
        batch_size=128,
        lr=0.002,
        b1=0.5,
        b2=0.999,
        n_epochs=200,
        sampling_interval=10
        )