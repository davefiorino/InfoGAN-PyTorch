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
dis_c = torch.zeros(100, 10, 1, 1, device=device) # tensor of zeros (100 x 10 x 1 x 1)
dis_c[torch.arange(0, 100), idx] = 1.0 
# Discrete latent code.
c = dis_c.view(100, -1, 1, 1) # tensor 100 x 10 x 1 x 1


zeros = torch.zeros(100, 10, 1, 1, device=device)
c1 = torch.cat((c, c, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros), dim=1)
c2 = torch.cat((c, zeros, c, zeros, zeros, zeros, zeros, zeros, zeros, zeros), dim=1)
c3 = torch.cat((c, zeros, zeros, c, zeros, zeros, zeros, zeros, zeros, zeros), dim=1)
c4 = torch.cat((c, zeros, zeros, zeros, c, zeros, zeros, zeros, zeros, zeros), dim=1)
c5 = torch.cat((c, zeros, zeros, zeros, zeros, c, zeros, zeros, zeros, zeros), dim=1)
c6 = torch.cat((c, zeros, zeros, zeros, zeros, zeros, c, zeros, zeros, zeros), dim=1)
c7 = torch.cat((c, zeros, zeros, zeros, zeros, zeros, zeros, c, zeros, zeros), dim=1)
c8 = torch.cat((c, zeros, zeros, zeros, zeros, zeros, zeros, zeros, c, zeros), dim=1)
c9 = torch.cat((c, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, c), dim=1)
codes = [c1, c2, c3, c4, c5, c6, c7, c8, c9]

z = torch.randn(100, 128, 1, 1, device=device) # random normal distributed values tensor (100 x 128 x 1 x 1)



# Generate images
i = 1
for code in codes:
    noise = torch.cat((z, code), dim=1) # 100 x 228 x 1 x 1
    with torch.no_grad():
        generated_img1 = netG(noise).detach().cpu()
    # Display the generated image.
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
    plt.savefig("output/generated_" + str(i))
    i += 1
    plt.show()

