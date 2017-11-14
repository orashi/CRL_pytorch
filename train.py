import argparse
import os
import random
import torch
from math import log10
import scipy.stats as stats
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable, grad
from models.crl import DispFulNet, DispResNet
from data.dataset import CreateFT3DLoader

parser = argparse.ArgumentParser()
parser.add_argument('--FT3D', type=str, default="/home/orashi/datasets/SFD", help='path to FlyingThings3D dataset')
parser.add_argument('--KITTI15', type=str, default="/home/orashi/datasets/???", help='path to KITTI 2015 dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)

parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cut', type=int, default=1, help='cut backup frequency')
parser.add_argument('--niter', type=int, default=700, help='number of epochs to train for')
parser.add_argument('--lrF', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--lrR', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netF', default='', help="path to netF (to continue training)")
parser.add_argument('--netR', default='', help="path to netR (to continue training)")
parser.add_argument('--optim', action='store_true', help='load optimizer\'s checkpoint')
parser.add_argument('--outf', default='.', help='folder to output images and models checkpoints')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--baseGeni', type=int, default=2500, help='start base of pure pair L1 loss')
parser.add_argument('--geni', type=int, default=0, help='continue gen image num')
parser.add_argument('--epoi', type=int, default=0, help='continue epoch num')
parser.add_argument('--env', type=str, default=None, help='tensorboard env')

opt = parser.parse_args()
print(opt)

####### regular set up
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
gen_iterations = opt.geni
try:
    os.makedirs(opt.outf)
except OSError:
    pass
# random seed setup                                  # !!!!!
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True
####### regular set up ends

writer = SummaryWriter(log_dir=opt.env)

dataloader = CreateFT3DLoader(opt)

netF = DispFulNet()
if opt.netF != '':
    netF.load_state_dict(torch.load(opt.netF))
print(netF)

netR = DispResNet()
if opt.netR != '':
    netR.load_state_dict(torch.load(opt.netR))
print(netR)

criterion_L1 = nn.L1Loss()
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netF.cuda()
    criterion_L1.cuda()

# setup optimizer
optimizerF = optim.Adam(netF.parameters(), lr=opt.lrF, betas=(opt.beta1, 0.9))
optimizerR = optim.Adam(netR.parameters(), lr=opt.lrR, betas=(opt.beta1, 0.9))
if opt.optim:
    optimizerF.load_state_dict(torch.load('%s/optimG_checkpoint.pth' % opt.outf))
    optimizerR.load_state_dict(torch.load('%s/optimD_checkpoint.pth' % opt.outf))

schedulerF = lr_scheduler.ReduceLROnPlateau(optimizerF, mode='max', verbose=True, min_lr=0.0000005,
                                            patience=8)  # 1.5*10^5 iter
schedulerR = lr_scheduler.ReduceLROnPlateau(optimizerR, mode='max', verbose=True, min_lr=0.0000005,
                                            patience=8)  # 1.5*10^5 iter
#schedulerG = lr_scheduler.MultiStepLR(optimizerG, milestones=[60, 120], gamma=0.1)  # 1.5*10^5 iter
#schedulerD = lr_scheduler.MultiStepLR(optimizerD, milestones=[60, 120], gamma=0.1)


flag = 1
for epoch in range(opt.epoi, opt.niter):

    epoch_loss = 0
    epoch_iter_count = 0

    for extra in range(2 * (opt.Diters + 1)):
        data_iter = iter(dataloader)
        iter_count = 0

        while iter_count < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netR.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            for p in netF.parameters():
                p.requires_grad = False  # to avoid computation

            # train the discriminator Diters times
            Diters = opt.Diters

            if gen_iterations < opt.baseGeni or not opt.adv:  # L1 stage
                Diters = 0

            j = 0
            while j < Diters and iter_count < len(dataloader):

                j += 1
                netD.zero_grad()

                real_bim, real_sim = data_iter.next()
                iter_count += 1

                if opt.cuda:
                    real_bim, real_sim = real_bim.cuda(), real_sim.cuda()

                # train with fake

                fake_sim = netF(Variable(real_bim, volatile=True)).data

                errD_fake = netD(Variable(torch.cat([fake_sim, real_bim], 1))).mean(0).view(1)
                errD_fake.backward(one, retain_graph=True)  # backward on score on real

                errD_real = netD(Variable(torch.cat([real_sim, real_bim], 1))).mean(0).view(1)
                errD_real.backward(mone)  # backward on score on real

                errD = errD_real - errD_fake

                # gradient penalty

                optimizerR.step()
            ############################
            # (2) Update G network
            ############################
            if iter_count < len(dataloader):

                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation
                for p in netF.parameters():
                    p.requires_grad = True  # to avoid computation
                netF.zero_grad()

                real_bim, real_sim = data_iter.next()
                iter_count += 1

                if opt.cuda:
                    real_bim, real_sim = real_bim.cuda(), real_sim.cuda()

                if flag:  # fix samples
                    writer.add_image('target imgs', vutils.make_grid(real_sim.mul(0.5).add(0.5), nrow=16))
                    writer.add_image('blur imgs', vutils.make_grid(real_bim.mul(0.5).add(0.5), nrow=16))
                    vutils.save_image(real_sim.mul(0.5).add(0.5),
                                      '%s/sharp_samples' % opt.outf + '.png')
                    vutils.save_image(real_bim.mul(0.5).add(0.5),
                                      '%s/blur_samples' % opt.outf + '.png')
                    fixed_blur = real_bim
                    flag -= 1

                fake = netF(Variable(real_bim))

                if gen_iterations < opt.baseGeni or not opt.adv:
                    contentLoss = criterion_L2(fake.mul(0.5).add(0.5), Variable(real_sim.mul(0.5).add(0.5)))
                    contentLoss.backward()

                    epoch_loss += 10 * log10(1 / contentLoss.data[0])
                    epoch_iter_count += 1
                    errG = contentLoss
                else:
                    errG = netD(torch.cat([fake, Variable(real_bim)], 1)).mean(0).view(1) * opt.advW
                    errG.backward(mone, retain_graph=True)

                    contentLoss = criterion_L2(fake.mul(0.5).add(0.5), Variable(real_sim.mul(0.5).add(0.5)))
                    contentLoss.backward()

                    epoch_loss += 10 * log10(1 / contentLoss.data[0])
                    epoch_iter_count += 1

                optimizerG.step()

            ############################
            # (3) Report & 100 Batch checkpoint
            ############################

            if gen_iterations < opt.baseGeni or not opt.adv:
                writer.add_scalar('MSE Loss', contentLoss.data[0], gen_iterations)
                print('[%d/%d][%d/%d][%d] err_G: %f'
                      % (epoch, opt.niter, iter_count + extra * len(dataloader_train),
                         len(dataloader_train) * 2 * (opt.Diters + 1), gen_iterations, contentLoss.data[0]))
            else:
                writer.add_scalar('MSE Loss', contentLoss.data[0], gen_iterations)
                writer.add_scalar('wasserstein distance', errD.data[0], gen_iterations)
                writer.add_scalar('errD_real', errD_real.data[0], gen_iterations)
                writer.add_scalar('errD_fake', errD_fake.data[0], gen_iterations)
                writer.add_scalar('Gnet loss toward real', errG.data[0], gen_iterations)
                writer.add_scalar('gradient_penalty', gradient_penalty.data[0], gen_iterations)
                print('[%d/%d][%d/%d][%d] errD: %f err_G: %f err_D_real: %f err_D_fake %f content loss %f'
                      % (epoch, opt.niter, iter_count + extra * len(dataloader_train),
                         len(dataloader_train) * 2 * (opt.Diters + 1),
                         gen_iterations, errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0],
                         contentLoss.data[0]))

            if gen_iterations % 100 == 0:
                fake = netF(Variable(fixed_blur, volatile=True))
                writer.add_image('deblur imgs', vutils.make_grid(fake.data.mul(0.5).add(0.5).clamp(0, 1), nrow=16),
                                 gen_iterations)

            if gen_iterations % 1000 == 0:
                for name, param in netF.named_parameters():
                    writer.add_histogram('netG ' + name, param.clone().cpu().data.numpy(), gen_iterations)
                for name, param in netD.named_parameters():
                    writer.add_histogram('netD ' + name, param.clone().cpu().data.numpy(), gen_iterations)
                vutils.save_image(fake.data.mul(0.5).add(0.5),
                                  '%s/fake_samples_gen_iter_%08d.png' % (opt.outf, gen_iterations))

            gen_iterations += 1

    if opt.test:
        if epoch % 5 == 0:
            avg_psnr = 0
            for batch in dataloader_test:
                input, target = [x.cuda() for x in batch]
                prediction = netF(Variable(input, volatile=True))
                mse = criterion_L2(prediction.mul(0.5).add(0.5), Variable(target.mul(0.5).add(0.5)))
                psnr = 10 * log10(1 / mse.data[0])
                avg_psnr += psnr
            avg_psnr = avg_psnr / len(dataloader_test)

            writer.add_scalar('Test epoch PSNR', avg_psnr, epoch)

            print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))

    avg_psnr = epoch_loss / epoch_iter_count
    writer.add_scalar('Train epoch PSNR', avg_psnr, epoch)
    schedulerG.step(avg_psnr)
    schedulerD.step(avg_psnr)

    # do checkpointing
    if opt.cut == 0:
        torch.save(netF.state_dict(), '%s/netG_epoch_only.pth' % opt.outf)
        torch.save(netD.state_dict(), '%s/netD_epoch_only.pth' % opt.outf)
    elif epoch % opt.cut == 0:
        torch.save(netF.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(optimizerG.state_dict(), '%s/optimG_checkpoint.pth' % opt.outf)
    torch.save(optimizerD.state_dict(), '%s/optimD_checkpoint.pth' % opt.outf)
