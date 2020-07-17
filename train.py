import argparse
import logging
import os
import sys
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from cv2 import *

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import DCM_Dataset
from torch.utils.data import DataLoader, random_split

dir_img = 'E:/Data_set/SAMPLE'
dir_checkpoint = 'checkpoints/'

def Trans_3Dlabel_2RGB(label, classes, is_pre=True):#label.shape = [C,H,W]
    RGB_img = np.zeros([3,label.shape[1],label.shape[2]])
    positins = []
    colors = [[0,0,0],[0,191,255],[46,139,87],[169,169,169],[124,252,0],[0,0,255],[255,0,255],[255,105,180],[255,165,0]]
    if is_pre:
        for i in range(0, classes):
            positins.append(label[i] == 1)
    else:
        for i in range(0, classes):
            positins.append(label[0] == i)
    for i in range(0, classes):
        RGB_img[0][positins[i]] = colors[i][0]#第i个class的位置
        RGB_img[1][positins[i]] = colors[i][1]
        RGB_img[2][positins[i]] = colors[i][2]
    return RGB_img

#TODO 取消bilinear *update:预计无影响，已恢复bilinear
def train_net(net,
              device,
              epochs=20,
              batch_size=1,
              lr=1e-5,
              val_percent=0.1,
              save_cp=True,
              img_scale=1.0):

    dataset = DCM_Dataset(dir_img)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print("TrainData length: ", len(train_loader), '\n', 'ValData length: ', len(val_loader))

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(12, epochs):
        net.train()
        if epoch>0 and os.path.exists(os.getcwd() + '/checkpoints/CP_epoch' + str(epoch) + '.pth'):
            net.load_state_dict(torch.load('./checkpoints/CP_epoch' + str(epoch) + '.pth'))
            print('\nTrain Parameter ' + str(epoch) + ' successfully loaded!')

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                # print('img', imgs.shape)
                # print('mask', true_masks.shape)
                # assert imgs.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                # assert imgs.shape[1] == true_masks.shape[1], 'Input 与 mask 的层数不同'

                # for i in range(imgs.shape[1]):
                optimizer.zero_grad()
                # img = torch.unsqueeze(imgs[:,i,:,:], dim=1)
                # true_mask = true_masks[:,i,:,:]
                # print(img.shape,true_mask.shape)

                masks_pred = net(imgs)

                # print("Dice:", Dice(true_mask.cpu().detach().numpy().astype(np.float16), (masks_pred[:,1,:,:]>0.5).cpu().detach().numpy().astype(np.float16)))

                # print(np.unique(img[0][0].cpu().detach().numpy()))
                # print(np.unique(true_mask[0].cpu().detach().numpy()))
                # print(np.unique(torch.sigmoid(masks_pred[0][1]).cpu().detach().numpy()>0.45))
                index = 3
                imshow("img",np.mat(((imgs[0][1].cpu().detach().numpy()+200)/400*255).astype(np.uint8)))
                #imshow("mask",np.mat((true_masks[0][true_masks[0]==index].cpu().detach().numpy()*255.0).astype(np.uint8)))
                imshow("pre",np.mat(((torch.sigmoid(masks_pred[0][index]).cpu().detach().numpy()>0.5)*
                                     255.0).astype(np.uint8)))
                waitKey(30)

                #print(np.unique((torch.sigmoid(masks_pred)>0.5).detach().cpu().numpy()))
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Loss/train', np.mean(loss.cpu().detach().numpy()), global_step)
                pbar.set_postfix(**{'loss (batch)': np.mean(loss.cpu().detach().numpy())})
                pbar.update(imgs.shape[0])
                del imgs

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        global_step += 1
        #print(global_step)
                # if global_step % (len(dataset) // (10 * batch_size)) == 0:
                #     val_score = eval_net(net, val_loader, device, n_val, epoch)#加载上一个epoch参数
                #     if net.n_classes > 1:
                #         logging.info('\nValidation cross entropy: {}'.format(val_score))
                #         writer.add_scalar('Loss/test', val_score, global_step)
                #
                #     else:
                #         logging.info('Validation Dice Coeff: {}'.format(val_score))
                #         writer.add_scalar('Dice/test', val_score, global_step)
                #
                #     writer.add_images('images', imgs, global_step)
                #     writer.add_images('images', true_mask*255, global_step)
                #     writer.add_images('images', torch.sigmoid(masks_pred[0][0])>0.5, global_step)
                #     #writer.add_images('masks/true', Trans_3Dlabel_2RGB(true_masks.cpu().detach().numpy(), 9, is_pre=False), global_step, dataformats='CHW')
                #     # labels = ['unkown','sky','tree','road','grass','water','building','mountain','foreground']
                #     # for i in range(0,9):
                #     #     translabel = torch.sigmoid(masks_pred[0][i]) > 0.5
                #     #     writer.add_images('masks/pred_'+ labels[i], translabel, 9, global_step, dataformats='HW')
                #
                #
                #         #print('show mask: ',str(i), show_mask.min(), show_mask.max())
                #     # show_mask[show_mask <= 0.5] = 0.0
                #     # masks_pred[0][1][masks_pred[0][1]>0.2]=1
                #
                #
                #     if net.n_classes == 1:
                #         writer.add_images('masks/true', true_mask, global_step)
                #         writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)



    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

def Dice(y_true, y_pred):
    y_true = y_true
    y_pred = y_pred
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=10, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')


    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    train_net(net,device=device)
    # # faster convolutions, but more memory
    # # cudnn.benchmark = True
    #
    # try:
    #     train_net(net=net,
    #               epochs=args.epochs,
    #               batch_size=args.batchsize,
    #               lr=args.lr,
    #               device=device,
    #               img_scale=args.scale,
    #               val_percent=args.val / 100)
    # except KeyboardInterrupt:
    #     torch.save(net.state_dict(), 'INTERRUPTED.pth')
    #     logging.info('Saved interrupt')
    #     try:
    #         sys.exit(0)
    #     except SystemExit:
    #         os._exit(0)
