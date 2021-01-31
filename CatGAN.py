# Using a dataset of 9000 pictures of cats (https://www.kaggle.com/crawford/cat-dataset),
# attempting to create a GAN that will generate new pictures of cats at 512x512 resolution.

import os
import random
import torch
import torch.nn as nn
#import torch.backends.cudnn as cudnn #Remove if on cpu-only installation
import torch.optim as optim
import torch.utils.data 
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

SIZE_Z = 100
G_FEATURE_SIZE = 64
D_FEATURE_SIZE = 64
IMAGE_SIZE = 512

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
            #size: 512 * 4 * 4
            nn.ConvTranspose2d(G_FEATURE_SIZE * 8, G_FEATURE_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_FEATURE_SIZE * 4),
            nn.ReLU(True),
            #size: 256 * 8 * 8
            nn.ConvTranspose2d(G_FEATURE_SIZE * 4, G_FEATURE_SIZE * 2, 8, 4, 2, bias=False),
            nn.BatchNorm2d(G_FEATURE_SIZE * 2),
            nn.ReLU(True),
            #size: 128 * 32 * 32
            nn.ConvTranspose2d(G_FEATURE_SIZE * 2,     G_FEATURE_SIZE, 8, 4, 2, bias=False),
            nn.BatchNorm2d(G_FEATURE_SIZE),
            nn.ReLU(True),
            #size: 64 * 128 * 128
            nn.ConvTranspose2d(    G_FEATURE_SIZE,                  3, 8, 4, 2, bias=False),
            nn.Tanh()
            #size: num_colors x 512 x 512
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #input is 3 x 512 x 512
            nn.Conv2d(                3,      D_FEATURE_SIZE, 8, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #size: 64 x 128 x 128
            nn.Conv2d(    D_FEATURE_SIZE, D_FEATURE_SIZE * 2, 8, 4, 2, bias=False),
            nn.BatchNorm2d(D_FEATURE_SIZE * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #size: 128 x 32 x 32
            nn.Conv2d(D_FEATURE_SIZE * 2, D_FEATURE_SIZE * 4, 8, 4, 2, bias=False),
            nn.BatchNorm2d(D_FEATURE_SIZE * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #size: 256 x 8 x 8
            nn.Conv2d(D_FEATURE_SIZE * 4, D_FEATURE_SIZE * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_FEATURE_SIZE * 8),
            nn.LeakyReLU(0.2, inplace=True),
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

    if manualSeed == None:
        manualSeed = random.randint(1, 10000)
    print("Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    '''
    if cuda:
        cudnn.benchmark = True
    '''

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
        previous_epoch = int(existing_G[-5])

    netD = Discriminator().to(device)
    netD.apply(weights_init)
    if existing_D != "":
        netD.load_state_dict(torch.load(existing_D))
        previous_epoch = int(existing_D[-5])

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, SIZE_Z, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    if dry_run:
        niter = 1

    for epoch in range(previous_epoch+1,previous_epoch+niter+1):
        running_errD = 0
        running_errG = 0
        running_D_x = 0
        running_D_G_z1 = 0
        running_D_G_z2 = 0
        for i, data in enumerate(dataloader, 1):
            #train with real image
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            running_D_x += output.mean().item()

            #train with fake image
            noise = torch.randn(batch_size, SIZE_Z, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            running_D_G_z1 += output.mean().item()
            running_errD += errD_real.item() + errD_fake.item()
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake,)
            errG = criterion(output, label)
            errG.backward()
            running_errG += errG.item()
            running_D_G_z2 += output.mean().item()
            optimizerG.step()
            if i == 1:
                print(f"Starting Epoch {epoch}...\n")
            elif i % 5 == 0:
                print(f"[{epoch}/{previous_epoch+niter}] [{i}/{len(dataloader)}]")
            elif i == len(dataloader):
                print(f'''
Completed Epoch {epoch}
Loss_D: {running_errD/i:.4f}
Loss_G: {running_errG/i:.4f} 
D(x): {running_D_x/i:.4f} 
D(G(z)): {running_D_G_z1/i:.4f}/{running_D_G_z2/i:.4f}
                ''')
                vutils.save_image(real_cpu, f"{outf}/Images/real_samples.png", normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(), f"{outf}/Images/fake_samples_epoch_{epoch}.png", normalize=True)
            if dry_run:
                break
    
    print("Run completed, ending execution")
    torch.save(netG.state_dict(), f"{outf}/netG_epoch_{epoch}.pth")
    torch.save(netD.state_dict(), f"{outf}/netD_epoch_{epoch}.pth")

if __name__ == '__main__':
    run_nn(cuda=False, niter=1)