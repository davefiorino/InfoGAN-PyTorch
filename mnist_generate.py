import argparse
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from models.mnist_model import Generator
from os import makedirs, path, mkdir
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
parser.add_argument('-s', '--save1k', action='store_true', help='Save 1000 figures to compute FID')
args = parser.parse_args()

# create directory to save output
if not path.isdir('output'):
    makedirs('output') 


# Load the checkpoint file
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator().to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
print(netG)

c = np.linspace(-2, 2, 10).reshape(1, -1)
c = np.repeat(c, 10, 0).reshape(-1, 1)
c = torch.from_numpy(c).float().to(device)
c = c.view(-1, 1, 1, 1)

zeros = torch.zeros(100, 1, 1, 1, device=device)

# Continuous latent code.
# c2 = torch.cat((c, zeros), dim=1)
# c3 = torch.cat((zeros, c), dim=1)
c2 = torch.cat((c, zeros), dim=1)
c3 = torch.cat((zeros, c), dim=1)

idx = np.arange(10).repeat(10)
dis_c = torch.zeros(100, 10, 1, 1, device=device)
dis_c[torch.arange(0, 100), idx] = 1.0
# Discrete latent code.
c1 = dis_c.view(100, -1, 1, 1)

# z = torch.randn(100, 62, 1, 1, device=device)
z = torch.randn(100, 62, 1, 1, device=device) 

# To see variation along c2 (Horizontally) and c1 (Vertically)
# noise1 = torch.cat((z, c1, c2), dim=1)
# To see variation along c3 (Horizontally) and c1 (Vertically)
# noise2 = torch.cat((z, c1, c3), dim=1)
noise1 = torch.cat((z, c1, c2), dim=1)
noise2 = torch.cat((z, c1, c3), dim=1)


# Generate image.
with torch.no_grad():
    generated_img1 = netG(noise1).detach().cpu()
# Display the generated image.
fig = plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.savefig("output/generated_1")
plt.show()

# Generate image.
with torch.no_grad():
    generated_img2 = netG(noise2).detach().cpu()
# Display the generated image.
fig = plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.savefig("output/generated_2")
plt.show()

# with torch.no_grad():
#     generated_img3 = netG(noise3).detach().cpu()
# # Display the generated image.
# fig = plt.figure(figsize=(10, 10))
# plt.axis("off")
# plt.imshow(np.transpose(vutils.make_grid(generated_img3, nrow=10, padding=2, normalize=True), (1,2,0)))
# plt.savefig("output/generated_3")
# plt.show()


if (args.save1k) == True:

    shutil.rmtree('output/imgs', ignore_errors=True)
    try: mkdir('output/imgs')
    except OSError: pass

    print("\nSaving 1000 figures...\n")
    # Generate and save 1000 images
    for i in range(5):
        # z = torch.randn(100, 62, 1, 1, device=device)
        z = torch.randn(100, 62, 1, 1, device=device) 
        # # To see variation along c2 (Horizontally) and c1 (Vertically)
        # noise1 = torch.cat((z, c1, c2), dim=1)
        # # To see variation along c3 (Horizontally) and c1 (Vertically)
        # noise2 = torch.cat((z, c1, c3), dim=1)
        noise1 = torch.cat((z, c1, c2), dim=1)
        noise2 = torch.cat((z, c1, c3), dim=1)

        with torch.no_grad():
            generated_img1 = netG(noise1).detach().cpu()

        for j in range(generated_img1.shape[0]):
            fig = plt.figure(figsize=(1, 1), dpi=28)
            plt.axis("off")
            plt.imshow((generated_img1[j])[0], cmap='gray')
            plt.savefig("output/imgs/generated_%d-1-%d" % (i, j), bbox_inches='tight', transparent="True", pad_inches=0)

        with torch.no_grad():
            generated_img2 = netG(noise2).detach().cpu()

        for j in range(generated_img2.shape[0]):
            fig = plt.figure(figsize=(1, 1), dpi=28)
            plt.axis("off")
            plt.imshow((generated_img2[j])[0], cmap='gray')
            plt.savefig("output/imgs/generated_%d-2-%d" % (i, j), bbox_inches='tight', transparent="True", pad_inches=0)

        # with torch.no_grad():
        #     generated_img3 = netG(noise3).detach().cpu()

        # for j in range(generated_img3.shape[0]):
        #     fig = plt.figure(figsize=(1, 1), dpi=28)
        #     plt.axis("off")
        #     plt.imshow((generated_img3[j])[0], cmap='gray')
        #     plt.savefig("output/imgs/generated_%d-3-%d" % (i, j), bbox_inches='tight', transparent="True", pad_inches=0)
