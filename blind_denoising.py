# coding=utf-8
#
#/*
# * Copyright (c) 2019, Thibaud Ehret <ehret.thibaud@gmail.com>
# * All rights reserved.
# *
# * This program is free software: you can use, modify and/or
# * redistribute it under the terms of the GNU Affero General Public
# * License as published by the Free Software Foundation, either
# * version 3 of the License, or (at your option) any later
# * version. You should have received a copy of this license along
# * this program. If not, see <http://www.gnu.org/licenses/>.
# */

import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from dncnn.models import *
import skimage
import skimage.io
import tifffile

from readFlowFile import read_flow
from scipy.ndimage.morphology import binary_dilation

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def psnr(img1, img2, peak=1):
    '''
    Computes the PSNR 'metric' between two images assumed to be in the range [0,1]
    '''
    x = ((np.array(img1).squeeze() - np.array(img2).squeeze()).flatten())
    return (10*np.log10(peak**2 / np.mean(x**2)))


class WarpedLoss(nn.Module):
    def __init__(self):
        super(WarpedLoss, self).__init__()
        self.criterion = nn.L1Loss(size_average=False)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow.
        Code heavily inspired by
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        grid = grid.cuda()
        vgrid = Variable(grid) + flo.cuda()

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :]/max(W-1, 1)-1.0
        vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :]/max(H-1, 1)-1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        # Define a first mask containing pixels that wasn't properly interpolated
        mask = torch.autograd.Variable(torch.ones(x.size()))
        mask = nn.functional.grid_sample(mask.cuda(), vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output, mask

    # Computes the occlusion map based on the optical flow
    def occlusion_mask(self, warped, of, old_mask):
        """
        Computes an occlusion mask based on the optical flow
        warped: [B, C, H, W] warped frame (only used for size)
        of: [B, 2, H, W] flow
        old_mask: [B, C, H, W] first estimate of the mask
        """
        a = np.zeros(warped.size())
        b = np.zeros(warped.size())

        # Compute an occlusion based on the divergence of the optical flow 
        a[:, :, :-1, :] = (of[0, 0, 1:, :] - of[0, 0, :-1, :])
        b[:, :, :, :-1] = (of[0, 1, :, 1:] - of[0, 1, :, :-1])
        mask = np.abs(a + b) > 0.75

        # Dilates slightly the occlusion map to be more conservative
        ball = np.zeros((3, 3))
        ball[1, 0] = 1
        ball[0, 1] = 1
        ball[1, 1] = 1
        ball[2, 1] = 1
        ball[1, 2] = 1
        mask[0, 0, :, :] = binary_dilation(mask[0, 0, :, :], ball)

        # Remove the boundaries (values extrapolated on the boundaries)
        mask[:, :, 0, :] = 1
        mask[:, :, mask.shape[2]-1, :] = 1
        mask[:, :, :, 0] = 1
        mask[:, :, :, mask.shape[3]-1] = 1

        # Invert the mask because we want a mask of good pixels
        mask = Variable((old_mask * torch.Tensor(1-mask).cuda()))
        return mask

    def forward(self, input, target, flow):
        # Warp input on target
        warped, mask = self.warp(target, flow)
        # Compute the occlusion mask
        mask = self.occlusion_mask(warped, flow, mask)
        # Compute the masked loss
        self.loss = self.criterion(mask*input, mask*warped)
        return self.loss


def blind_denoising(**args):
    """
    Main function
    args: Parameters
    """

    ################
    # LOAD THE MODEL
    ################

    model_fn = args['network']
    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            model_fn)

    # By default the code is configured to load one of the pre-trained DnCNN.
    # The code needs to be adapted to be used with other networks
    if 1:
        print('Loading model a pre-trained DnCNN\n')
        net = DnCNN(channels=1, num_of_layers=17)
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        model.load_state_dict(torch.load(model_fn))
    else:
        print('Loading model already fine-tuned\n')
        model = torch.load(model_fn)[0]
        model.cuda()

    #################
    # DEFINE THE LOSS
    #################

    # The loss needs to be changed when used with different networks
    lr = 5e-5
    weight_decay = 0.00001

    criterion = WarpedLoss()
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=weight_decay, amsgrad=False)

    ###########
    # MAIN LOOP
    ###########
    if args['input'][-4:] == 'tiff' or args['input'][-3:] == 'tif':
        imorig = tifffile.imread(args['input'] % (args['first']))
    else:
        imorig = skimage.io.imread(
            args['input'] % (args['first']), as_gray=True)

    # If the input data has not been preprocessed then put it in batch style and normalize the values
    if len(imorig.shape) < 4:
        imorig = np.expand_dims(imorig, 0)
        imorig = np.expand_dims(imorig, 0)
        imorig = imorig/255.

    prev_frame_var = Variable(torch.Tensor(imorig).cuda())

    # Write the psnr per frame in this file
    plot_psnr = open(args['output_psnr'], 'w')

    for i in range(args['first']+1, args['last']+1):
        # Print the index of the frame to track the progress
        print(i)

        # Load new frame
        if args['input'][-4:] == 'tiff' or args['input'][-3:] == 'tif':
            curr_frame = tifffile.imread(args['input'] % (i))
        else:
            curr_frame = skimage.io.imread(args['input'] % (i), as_gray=True)

        # If the input data has not been preprocessed then put it in batch style and normalize the values
        if len(curr_frame.shape) < 4:
            curr_frame = np.expand_dims(curr_frame, 0)
            curr_frame = np.expand_dims(curr_frame, 0)
            curr_frame = curr_frame/255.

        curr_frame_var = Variable(torch.FloatTensor(curr_frame).cuda())

        # Load optical flow
        flow = read_flow(args['flow'] % (i))
        flow = np.expand_dims(flow, 0)
        flow = Variable(torch.Tensor(flow))
        flow = flow.permute(0, 3, 1, 2)

        # Set the network to training mode
        model.train()
        optimizer.zero_grad()

        # Do noise2noise1shot learning
        for it in range(args['iter']):
            #start_time = time.time()
            out_train = curr_frame_var - model(curr_frame_var)
            loss = criterion(out_train, prev_frame_var, flow)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print("time for one iteration ", time.time()-start_time)

        # compute and save the denoised
        model.eval()

        # Denoise the frame
        with torch.no_grad():  # PyTorch v0.4.0
            im_noise_estim = curr_frame_var - model(curr_frame_var)

        # Write output images
        if args['ref'][-4:] == 'tiff' or args['ref'][-3:] == 'tif':
            tifffile.imsave(args['output'] % (i), np.squeeze(
                255.*np.array(im_noise_estim.cpu())))
        else:
            skimage.io.imsave(args['output'] % (i), np.squeeze(
                255.*np.clip(np.array(im_noise_estim.cpu()), 0., 1.)).astype(np.uint8))

        # Load new frame
        if args['ref'][-4:] == 'tiff' or args['ref'][-3:] == 'tif':
            ref_frame = tifffile.imread(args['ref'] % (i))
        else:
            ref_frame = skimage.io.imread(args['ref'] % (i), as_gray=True)
        if len(ref_frame.shape) < 4:
            ref_frame = np.expand_dims(ref_frame, 0)
            ref_frame = np.expand_dims(ref_frame, 0)
            ref_frame = ref_frame/255.

        # Compute the PSNR according to the reference frame
        quant = psnr(ref_frame, im_noise_estim.cpu())
        plot_psnr.write(str(quant)+'\n')
        print(quant)

        # Move to the next frame
        prev_frame_var = curr_frame_var

    torch.save([model, optimizer], args['output_network'])
    plot_psnr.close()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Blind_denoising_grayscale")
    parser.add_argument("--input", type=str, default="",
                        help='path to input frames (C type)')
    parser.add_argument("--ref",   type=str, default="",
                        help='path to reference frames (C type), against which the psnr is going to be computed')
    parser.add_argument("--flow", type=str, default="",
                        help='path to optical flow (C type)')
    parser.add_argument("--output", type=str, default="./%03d.png",
                        help='path to output image (C type)')

    parser.add_argument("--output_psnr", type=str, default="plot_psnr.txt",
                        help='path to output psnr')
    parser.add_argument("--output_network", type=str, default="final.pth",
                        help='path to output network')

    parser.add_argument("--first", type=int, default=1,
                        help='index first frame')
    parser.add_argument("--last", type=int, default=300,
                        help='index last frame')

    parser.add_argument("--iter", type=int, default=20,
                        help='number of time the learning is done on a given frame')

    parser.add_argument("--network", type=str, default="dncnn/logs/DnCNN-S-25/net.pth",
                        help='path to the network')

    argspar = parser.parse_args()

    print("\n### Model-blind Video Denoising Via Frame-to-frame Training ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    blind_denoising(**vars(argspar))
