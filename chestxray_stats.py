import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets

dataset = datasets.ImageFolder(root='data/chest_xray/train', transform=transforms.Compose([transforms.Resize(256),
                             transforms.CenterCrop(256),
                             transforms.ToTensor()]))

loader = data.DataLoader(dataset,
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
std = torch.sqrt(var / (len(loader.dataset)*224*224))
print(std)