import argparse
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from models.chestxray_model import Generator
from config import params
from os import makedirs, path

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
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



c = np.linspace(-2, 2, 10).reshape(1, -1) # 10 evenly spaced numbers between -2 and 2. (1 x 10 vector) 
c = np.repeat(c, 10, 0).reshape(-1, 1) # repeat each number in the array 10 times. (100 x 1 vector)
c = torch.from_numpy(c).float().to(device)
c = c.view(-1, 1, 1, 1) # tensor 100 x 1 x 1 x 1

zeros = torch.zeros(100, 1, 1, 1, device=device) # tensor of zeros (100 x 1 x 1 x 1)

# Continuous latent code.
c2 = torch.cat((c, zeros), dim=1) # concatenate c and zeros (100 x 2 x 1 x 1)
c3 = torch.cat((zeros, c), dim=1) # concatenate c and zeros (100 x 2 x 1 x 1)


idx = np.arange(10).repeat(10) # integers from 0 to 9, each repeated 10 times
dis_c = torch.zeros(100, 10, 1, 1, device=device) # tensor of zeros (100 x 10 x 1 x 1)
dis_c[torch.arange(0, 100), idx] = 1.0
# Discrete latent code.
c1 = dis_c.view(100, -1, 1, 1) # tensor 100 x 10 x 1 x 1

z = torch.randn(100, 100, 1, 1, device=device) # random normal distributed values tensor (100 x 100 x 1 x 1) The second 100 is num_z


# To see variation along c2-c3 (Horizontally) and c1 (Vertically)
noise1 = torch.cat((z, c1, c2), dim=1) # 100 x 112 x 1 x 1
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
