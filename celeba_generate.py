import argparse
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from models.celeba_model import Generator
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



idx = np.arange(10).repeat(10) # integers from 0 to 9, each repeated 10 times
dis_c = torch.zeros(100, 10, 10, 1, device=device) # tensor of zeros (100 x 10 x 10 x 1)

for i in [0,1]: # indexes of codes that you want to vary from 0 to 9
    dis_c[torch.arange(0, 100), i, idx] = 1.0 

idx2 = np.zeros(100)
for i in [2,3,4,5,6,7,8,9]: # indexes of codes that you want to keep fixed at 0
    dis_c[torch.arange(0, 100), i, idx2] = 1.0 


# Discrete latent code.
c1 = dis_c.view(100, -1, 1, 1) # tensor 100 x 100 x 1 x 1




z = torch.randn(100, 128, 1, 1, device=device) # random normal distributed values tensor (100 x 128 x 1 x 1)

# Generate images

noise = torch.cat((z, c1), dim=1) # 100 x 228 x 1 x 1
with torch.no_grad():
    generated_img1 = netG(noise).detach().cpu()
# Display the generated image.
fig = plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.savefig("output/generated_1")
plt.show()

