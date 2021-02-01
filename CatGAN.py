# Using a dataset of 9000 pictures of cats (https://www.kaggle.com/crawford/cat-dataset),
# attempting to create a GAN that will generate new pictures of cats at 512x512 resolution.

import os
import random
import time
import re
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn #Remove if on cpu-only installation
import torch.optim as optim
import torch.utils.data 
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

SIZE_Z = 100
G_FEATURE_SIZE = 64
D_FEATURE_SIZE = 64
IMAGE_SIZE = 128

device = "cpu"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            #Z is latent vector of noise
            nn.ConvTranspose2d(            SIZE_Z, G_FEATURE_SIZE * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_FEATURE_SIZE * 8),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5, inplace=True),
            #size: 512 * 4 * 4
            nn.ConvTranspose2d(G_FEATURE_SIZE * 8, G_FEATURE_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_FEATURE_SIZE * 4),
            nn.ReLU(True),
            nn.Dropout2d(0.2, True),
            #size: 256 * 8 * 8
            nn.ConvTranspose2d(G_FEATURE_SIZE * 4, G_FEATURE_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_FEATURE_SIZE * 2),
            nn.ReLU(True),
            nn.Dropout2d(0.2, True),
            #size: 128 * 16 * 16
            nn.ConvTranspose2d(G_FEATURE_SIZE * 2,     G_FEATURE_SIZE, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_FEATURE_SIZE),
            nn.ReLU(True),
            nn.Dropout2d(0.2, True),
            #size: 64 * 32 * 32
            nn.ConvTranspose2d(    G_FEATURE_SIZE,                  3, 8, 4, 2, bias=False),
            nn.Tanh()
            #size: num_colors x 128 x 128
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #input is 3 x 128 x 128
            nn.Conv2d(                3,      D_FEATURE_SIZE, 8, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5, True),
            #size: 64 x 32 x 32
            nn.Conv2d(    D_FEATURE_SIZE, D_FEATURE_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_FEATURE_SIZE * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2, True),
            #size: 128 x 16 x 16
            nn.Conv2d(D_FEATURE_SIZE * 2, D_FEATURE_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_FEATURE_SIZE * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2, True),
            #size: 256 x 8 x 8
            nn.Conv2d(D_FEATURE_SIZE * 4, D_FEATURE_SIZE * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_FEATURE_SIZE * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2, True),
            #size: 512 x 4 x 4
            nn.Conv2d(D_FEATURE_SIZE * 8,                  1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            #size: 1
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

def run_nn(workers=2, batch_size=64, niter=25, lr=0.0002, beta1=0.5, 
           cuda=True, 
           dry_run=False, 
           existing_G="", 
           existing_D="", 
           outf="./CatGAN/Output", 
           manualSeed=None):
    try:
        os.makedirs(outf)
    except OSError:
        pass

    try:
        os.makedirs(outf + "/Images")
    except OSError:
        pass

    if manualSeed == None:
        manualSeed = random.randint(1, 10000)
    print("Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    torch.autograd.set_detect_anomaly(True)

    if cuda:
        cudnn.benchmark = True
    

    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with cuda=True")

    #folder dataset
    dataset = dset.ImageFolder(root="./CatGAN/data",
                               transform=transforms.Compose([
                                   transforms.Resize(IMAGE_SIZE),
                                   transforms.CenterCrop(IMAGE_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5, 0.5))
                               ]))

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=int(workers))

    device = torch.device("cuda:0" if cuda else "cpu")

    netG = Generator().to(device)
    netG.apply(weights_init)
    previous_epoch = 0
    if existing_G != "":
        netG.load_state_dict(torch.load(existing_G))
        previous_epoch = int(re.search(r'\d+', existing_G).group())

    netD = Discriminator().to(device)
    netD.apply(weights_init)
    if existing_D != "":
        netD.load_state_dict(torch.load(existing_D))
        previous_epoch = int(re.search(r'\d+', existing_D).group())

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, SIZE_Z, 1, 1, device=device)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, 15, 0.5)
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, 15, 0.5)

    if dry_run:
        niter = 1

    for epoch in range(previous_epoch+1,previous_epoch+niter+1):
        start_time = time.time()
        running_errD = 0
        running_errG = 0
        running_D_x = 0
        running_D_G_z1 = 0
        running_D_G_z2 = 0
        for i, data in enumerate(dataloader, 1):
            #send data to device
            real_images = data[0].to(device)
            batch_size = real_images.size(0)

            #tensor of ones with size=batch_size for loss computation
            batch_size_ones = torch.full((batch_size,), 1, dtype=real_images.dtype, device=device)
            #tensor of zeroes with size=batch_size for loss computation
            batch_size_zeroes = torch.full((batch_size,), 0, dtype=real_images.dtype, device=device)
            #tensor of noise with size=batch_size for image generation
            batch_size_noise = torch.randn(batch_size, SIZE_Z, 1, 1, device=device)
            

            #create a batch of fake images
            fake_images = netG(batch_size_noise)

            #get likelihood that fake images are real
            G_fake_is_real = netD(fake_images,)
            Loss_G = criterion(G_fake_is_real, batch_size_ones)

            #optimize the generator
            optimizerG.zero_grad()
            Loss_G.backward()
            optimizerG.step()

            D_real_is_real = netD(real_images)
            D_fake_is_real = netD(fake_images.detach())
            loss_D_real = criterion(D_real_is_real, batch_size_ones)
            loss_D_fake = criterion(D_fake_is_real, batch_size_zeroes)

            #optimize the discriminator
            optimizerD.zero_grad()
            loss_D_real.backward()
            loss_D_fake.backward()
            optimizerD.step()

            

            #Add to running totals of errors
            running_errD += loss_D_real.item() + loss_D_fake.item()
            running_errG += Loss_G.item()
            running_D_x += D_real_is_real.mean().item()
            running_D_G_z1 += D_fake_is_real.mean().item()
            running_D_G_z2 += G_fake_is_real.mean().item()

            if i == 1:
                print(f"Starting Epoch {epoch}...\n")
            elif i % 40 == 0:
                print(f"[{epoch}/{previous_epoch+niter}] [{i}/{len(dataloader)}]")
            elif i == len(dataloader):
                run_time = time.time() - start_time
                print(f'''
Completed Epoch {epoch}
Loss_D: {running_errD/i:.4f}
Loss_G: {running_errG/i:.4f} 
D(x): {running_D_x/i:.4f} 
D(G(z)): {running_D_G_z1/i:.4f}/{running_D_G_z2/i:.4f}
Time to complete: {run_time:.2f} seconds
                ''')
            if dry_run:
                break

        #Execute at the end of each epoch    
        vutils.save_image(real_images, f"{outf}/Images/real_samples.png", normalize=True)
        fake_images = netG(fixed_noise)
        vutils.save_image(fake_images.detach(), f"{outf}/Images/fake_samples_epoch_{epoch}.png", normalize=True)
        schedulerD.step()
        schedulerG.step()
        if (epoch-1) % 15 == 0:
            print(f"New lr: {schedulerD.get_last_lr}")
    
    print("Run completed, ending execution")
    torch.save(netG.state_dict(), f"{outf}/netG_epoch_{epoch}.pth")
    torch.save(netD.state_dict(), f"{outf}/netD_epoch_{epoch}.pth")

if __name__ == '__main__':
    run_nn(cuda=True,
           niter=60, 
           lr=0.0005)