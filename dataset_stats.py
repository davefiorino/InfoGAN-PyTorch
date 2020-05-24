# Compute mean and std of the dataset
# ChestXRay: mean=0.5711, std=0.1774 

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets

dataset = dsets.ImageFolder(root='data/chest_xray/train', transform=transforms.Compose([transforms.Resize(256),
                             transforms.CenterCrop(256),
                             transforms.ToTensor()]))

loader = torch.utils.data.DataLoader(dataset,
                        batch_size=128,
                         num_workers=0,
                         shuffle=False)

mean = 0.0
for images, _ in loader:
    batch_samples = images.size(0) 
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
mean = mean / len(loader.dataset)
print(mean)

var = 0.0
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1))**2).sum([0,2])
std = torch.sqrt(var / (len(loader.dataset)*256*256))
print(std)







