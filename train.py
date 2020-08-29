import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random
from models.mnist_model import Generator, Discriminator, DHead, QHead
from dataloader import get_data
from utils import *
import config
from config import params
from mytorchsummary import summary
import argparse
from os import makedirs, path
from shutil import copyfile, rmtree

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=False, help='Checkpoint path to resume training')
parser.add_argument('-reset_output', '-ro', required=False, action='store_true', help='Add this argument to clean the output folder')
args = parser.parse_args()

config.init()

# create directory to save output
if args.reset_output and path.isdir('output'):
    rmtree('output')
try: 
    makedirs('output') 
    makedirs('output/checkpoint') 
except OSError as error: 
    if error.errno == 17:
        print("ERROR 17: Directory 'output' alredy exists in paht, remove it or use the option -ro to delete it.")
    else:
        print(error)
    exit()

if(params['dataset'] == 'MNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'SVHN'):
    from models.svhn_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'CelebA'):
    from models.celeba_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'FashionMNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'ChestXRay'):
    from models.chestxray_model import Generator, Discriminator, DHead, QHead

# Set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Use GPU if available.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

dataloader = get_data(params['dataset'], params['batch_size'])
print("Dataset element shape: ", dataloader.dataset[0][0].shape, '\n')

# Set appropriate hyperparameters depending on the dataset used.
# The values given in the InfoGAN paper are used.
# num_z : dimension of incompressible noise.
# num_dis_c : number of discrete latent code used.
# dis_c_dim : dimension of discrete latent code.
# num_con_c : number of continuous latent code used.
if(params['dataset'] == 'MNIST'):
    params['num_z'] = 62 
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2 
elif(params['dataset'] == 'SVHN'):
    params['num_z'] = 124
    params['num_dis_c'] = 4
    params['dis_c_dim'] = 10
    params['num_con_c'] = 4
elif(params['dataset'] == 'CelebA'):
    params['num_z'] = 128
    params['num_dis_c'] = 10
    params['dis_c_dim'] = 10
    params['num_con_c'] = 0
elif(params['dataset'] == 'FashionMNIST'):
    params['num_z'] = 62 #62 
    params['num_dis_c'] = 0
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2
elif(params['dataset'] == 'ChestXRay'):
    params['num_z'] = 100
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2

# Plot the training images.
sample_batch = next(iter(dataloader))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 100], nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.savefig('output/Training Images {}'.format(params['dataset']))
plt.close('all')

# Initialise the network.
netG = Generator().to(device)
netG.apply(weights_init)
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
netD = DHead().to(device)
netD.apply(weights_init)
netQ = QHead().to(device)
netQ.apply(weights_init)

# Print model summary
logFile = open("output/log.txt", "a")
if params['print_model_description']:
    print(netG)
    logFile.write(netG.__repr__() + '\n')
    noise_shape = noise_sample(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'], params['batch_size'], device)[0].shape
    summary(netG, input_size=(noise_shape[1], noise_shape[2], noise_shape[3]), log_file=logFile)
    print(discriminator)
    logFile.write(discriminator.__repr__() + '\n')
    dataset_shape = dataloader.dataset[0][0].shape
    summary(discriminator, input_size=(dataset_shape[0], dataset_shape[1], dataset_shape[2]), log_file=logFile)
    print(netD)
    logFile.write(netD.__repr__() + '\n')
    #summary(netD, input_size=(1024, 4, 4), log_file=logFile) # Must set the right input size manually
    print(netQ)
    logFile.write(netQ.__repr__() + '\n')
    #summary(netQ, input_size=(1024, 4, 4), log_file=logFile) # Must set the right input size manually

# Loss for discrimination between real and fake images.
criterionD = nn.BCELoss()
# Loss for discrete latent code.
criterionQ_dis = nn.CrossEntropyLoss()
# Loss for continuous latent code.
criterionQ_con = NormalNLLLoss()

# Adam optimiser is used.
optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}],lr=params['learning_rate_D'], betas=(params['beta1'], params['beta2']))
optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=params['learning_rate_G'], betas=(params['beta1'], params['beta2']))

# Resume training
if args.load_path:
    checkpoint = torch.load(args.load_path)
    netG.load_state_dict(checkpoint['netG'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    netD.load_state_dict(checkpoint['netD'])
    netQ.load_state_dict(checkpoint['netQ'])
    optimD.load_state_dict(checkpoint['optimD'])
    optimG.load_state_dict(checkpoint['optimG'])

# Fixed Noise  (comment dimensions are for MNIST)
z = torch.randn(100, params['num_z'], 1, 1, device=device) # tensor of random numebers. 100 x 62 x 1 x 1
fixed_noise = z
if(params['num_dis_c'] != 0):
    idx = np.arange(params['dis_c_dim']).repeat(10) # array of numbers from 0 to 10, repeated 10 times [0,0,0,0(x10) .... ,1,1,1,1(x10)... 9] (100 elements)
    dis_c = torch.zeros(100, params['num_dis_c'], params['dis_c_dim'], device=device) # tensor of zeros. 100 x 1 x 10
    for i in range(params['num_dis_c']): # i = 0
        dis_c[torch.arange(0, 100), i, idx] = 1.0 # disc_c[ [0,...99] , 0 , idx ] = 1.0
    dis_c = dis_c.view(100, -1, 1, 1) # 100 x 10 x 1 x 1
    fixed_noise = torch.cat((fixed_noise, dis_c), dim=1) 
if(params['num_con_c'] != 0):
    con_c = torch.rand(100, params['num_con_c'], 1, 1, device=device) * 2 - 1  # 100 x 2 x 1 x 1
    fixed_noise = torch.cat((fixed_noise, con_c), dim=1)

real_label = 1
fake_label = 0

# List variables to store results pf training.
img_list = []
G_losses = []
D_losses = []

print("-"*25)
print("Starting Training Loop...")
print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (params['num_epochs'], params['batch_size'], len(dataloader)))
print("-"*25)
logFile.write("-"*25)
logFile.write("\nStarting Training Loop...\n")
logFile.write('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d \n'.format(params['dataset']) % (params['num_epochs'], params['batch_size'], len(dataloader)))
logFile.write('Learning rate D: %f\nLearning rate G: %f\nBeta 1: %f\nBeta 2: %f\n' % (params['learning_rate_D'], params['learning_rate_G'], params['beta1'], params['beta2']))
logFile.write('Dropout: %r\n' % (params['use_dropout']))
logFile.write("-"*25)
logFile.close()

start_time = time.time()
iters = 0

for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()
    config.currentEpoch = epoch

    for i, (data, _) in enumerate(dataloader, 0):
        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)

        # Updating discriminator and DHead
        optimD.zero_grad()
        # Real data
        label = torch.full((b_size, ), real_label, device=device, dtype=torch.float32)
        output1 = discriminator(real_data)
        probs_real = netD(output1).view(-1)
        loss_real = criterionD(probs_real, label)
        # Calculate gradients.
        loss_real.backward()

        # Fake data
        label.fill_(fake_label)
        noise, idx = noise_sample(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'], b_size, device)
        fake_data = netG(noise)
        output2 = discriminator(fake_data.detach())
        probs_fake = netD(output2).view(-1)
        loss_fake = criterionD(probs_fake, label)
        # Calculate gradients.
        loss_fake.backward()

        # Net Loss for the discriminator
        D_loss = loss_real + loss_fake
        # Update parameters
        optimD.step()

        # Updating Generator and QHead
        optimG.zero_grad()

        # Fake data treated as real.
        output = discriminator(fake_data)
        label.fill_(real_label)
        probs_fake = netD(output).view(-1)
        gen_loss = criterionD(probs_fake, label)

        q_logits, q_mu, q_var = netQ(output)
        target = torch.LongTensor(idx).to(device)
        # Calculating loss for discrete latent code.
        dis_loss = 0
        for j in range(params['num_dis_c']):
            dis_loss += criterionQ_dis(q_logits[:, j*10 : j*10 + 10], target[j])

        # Calculating loss for continuous latent code.
        con_loss = 0
        if (params['num_con_c'] != 0):
            con_loss = criterionQ_con(noise[:, params['num_z']+ params['num_dis_c']*params['dis_c_dim'] : ].view(-1, params['num_con_c']), q_mu, q_var)*0.1

        # Net loss for generator.
        G_loss = gen_loss + dis_loss + con_loss
        # Calculate gradients.
        G_loss.backward()
        # Update parameters.
        optimG.step()

        # Check progress of training.
        if i != 0 and i%10 == 0:
            print('\n[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tdis_loss: %.4f  con_loss: %.4f'
                  % (epoch+1, params['num_epochs'], i, len(dataloader), 
                    D_loss.item(), G_loss.item(), dis_loss, con_loss))
            logFile = open("output/log.txt", "a")
            logFile.write('\n[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tdis_loss: %.4f\tcon_loss: %.4f'
                  % (epoch+1, params['num_epochs'], i, len(dataloader), 
                    D_loss.item(), G_loss.item(), dis_loss, con_loss))
            logFile.close()

        # Save the losses for plotting.
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

        iters += 1

    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
    logFile = open("output/log.txt", "a")
    logFile.write("\nTime taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
    logFile.close()
    # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
    with torch.no_grad():
        gen_data = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))

    # Generate image to check performance of generator.
    if (epoch+1) == 1 or (epoch+1) % params['image_log_epoch'] == 0:
        with torch.no_grad():
            gen_data = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.savefig("output/Epoch_%d {}".format(params['dataset']) %(epoch+1))
        plt.close('all')

    # Save network weights.
    if (epoch+1) % params['save_epoch'] == 0:
        torch.save({
            'netG' : netG.state_dict(),
            'discriminator' : discriminator.state_dict(),
            'netD' : netD.state_dict(),
            'netQ' : netQ.state_dict(),
            'optimD' : optimD.state_dict(),
            'optimG' : optimG.state_dict(),
            'params' : params
            }, 'output/checkpoint/model_epoch_%d_{}'.format(params['dataset']) %(epoch+1))

training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
print("-"*50)
logFile = open("output/log.txt", "a")
logFile.write('\n')
logFile.write("-"*50)
logFile.write('\nTraining finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
logFile.close()


# Save network weights.
torch.save({
    'netG' : netG.state_dict(),
    'discriminator' : discriminator.state_dict(),
    'netD' : netD.state_dict(),
    'netQ' : netQ.state_dict(),
    'optimD' : optimD.state_dict(),
    'optimG' : optimG.state_dict(),
    'params' : params
    }, 'output/checkpoint/model_final_{}'.format(params['dataset']))

# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("output/Loss Curve {}".format(params['dataset']))

# Animation showing the improvements of the generator.
# bug with matplotlib 3.2.1 https://github.com/matplotlib/matplotlib/issues/17097
# fig = plt.figure(figsize=(10,10))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
# anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# anim.save('output/infoGAN_{}.gif'.format(params['dataset']), dpi=80, writer='imagemagick')
# plt.show()
