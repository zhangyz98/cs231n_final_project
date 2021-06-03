from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pdb
from logger import Logger
import os
import numpy as np
from PIL import Image

############## Image related
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
####################

### data augmentation / Yizhi 0520
###
### separated coarse and fine net to accommodate up-sampling / Yizhi 0521
### 0521 version directly overwrites the previous version since not many changes have been made here.
### To recover: 1) change all loaders following the commented data loader block (name and depth_transform)
###   (0521)    2) change constants: output_height and output_width back
###             3) current optimizer is Adam to accommodate added layers, could change back to SGD
###             4) in all four optimization processes (train_coarse, train_fine, val_coarse, val_fine), change the name of data loader back ###                to the commented line

# Training settings
parser = argparse.ArgumentParser(description='PyTorch depth map prediction example')
parser.add_argument('model_folder', type=str, metavar='F',
                    help='In which folder do you want to save the model')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type = int, default = 32, metavar = 'N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default = 10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--suffix', type=str, default='', metavar='D',
                    help='suffix for the filename of models and output files')
args = parser.parse_args()

torch.manual_seed(args.seed)    # setting seed for random number generation

### size * 2: Trying up-sampling layer / Yizhi 0521
output_height = 55 * 2 #55 
output_width = 74 * 2 #74

# output_height = 55
# output_width = 74

# save loss and accuracy values modified by Yizhi (0511)
# trainig loss
coarse_train_loss_hist = []
fine_train_loss_hist = []
# validation loss
coarse_val_loss_hist = []
fine_val_loss_hist = []
# validation accuracy (delat3)
coarse_val_acc_hist = []
fine_val_acc_hist = []
# validation error (rmse linear)
coarse_val_rmse_lin_hist = []
fine_val_rmse_lin_hist = []
# validation error (rmse log)
coarse_val_rmse_log_hist = []
fine_val_rmse_log_hist = []


# accumulate err modified by zz 0511
coarse_val_accum_err = []
prev_coarse_val_accum = 0
fine_val_accum_err = []
prev_fine_val_accum = 0



date = args.model_folder #'05081225'
txt_out_dir = 'txt_results/' + date + '/'
# txt_out_dir = 'txt_results/' + date + '_wo_bn'
# img_out_dir = 'img_results/' + date
if not os.path.exists(txt_out_dir): os.mkdir(txt_out_dir)
# if not os.path.exists(img_out_dir): os.mkdir(img_out_dir)

from data import NYUDataset, NYUDataset_train, rgb_data_transforms, depth_data_transforms_coarse, depth_data_transforms_fine
from image_helper import plot_grid



'''# example data loader if not using upsampling
transformed_train_loader = NYUDataset_train('nyu_depth_v2_labeled.mat',
                                            'image_folder/training_image_',
                                            'depth_folder/training_depth_',
                                            'training',
                                            rgb_transform = rgb_data_transforms,
                                            depth_transform = depth_data_transforms)
'''

transformed_train_loader_coarse = NYUDataset_train('nyu_depth_v2_labeled.mat',
                                            'image_folder/training_image_',
                                            'depth_folder/training_depth_',
                                            'training',
                                            rgb_transform = rgb_data_transforms,
                                            depth_transform = depth_data_transforms_coarse)

transformed_train_loader_fine = NYUDataset_train('nyu_depth_v2_labeled.mat',
                                            'image_folder/training_image_',
                                            'depth_folder/training_depth_',
                                            'training',
                                            rgb_transform = rgb_data_transforms,
                                            depth_transform = depth_data_transforms_fine)

train_loader_coarse = torch.utils.data.DataLoader(transformed_train_loader_coarse, 
                                           batch_size = args.batch_size, 
                                           # changed num_workers:
                                           shuffle = True, num_workers = 2)
                                           # shuffle = True, num_workers = 5)
    
train_loader_fine = torch.utils.data.DataLoader(transformed_train_loader_fine, 
                                           batch_size = args.batch_size, 
                                           # changed num_workers:
                                           shuffle = True, num_workers = 2)
                                           # shuffle = True, num_workers = 5)

val_loader_coarse = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat',
                                                       'validation', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms_coarse), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 2)
                                            # shuffle = False, num_workers = 5)

    
val_loader_fine = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat',
                                                       'validation', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms_fine), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 2)
                                            # shuffle = False, num_workers = 5)
    
test_loader_coarse = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat',
                                                       'test', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms_coarse), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 0)
                                            # shuffle = False, num_workers = 5)
    
test_loader_fine = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat',
                                                       'test', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms_fine), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 0)
                                            # shuffle = False, num_workers = 5)


from model import coarseNet, fineNet
coarse_model = coarseNet()
fine_model = fineNet()
coarse_model.cuda()
fine_model.cuda()

# Paper values for SGD
# coarse_optimizer = optim.SGD([{'params': coarse_model.conv1.parameters(), 'lr': 0.001},{'params': coarse_model.conv2.parameters(), 'lr': 0.001},{'params': coarse_model.conv3.parameters(), 'lr': 0.001},{'params': coarse_model.conv4.parameters(), 'lr': 0.001},{'params': coarse_model.conv5.parameters(), 'lr': 0.001},{'params': coarse_model.fc1.parameters(), 'lr': 0.1},{'params': coarse_model.fc2.parameters(), 'lr': 0.1}], lr = 0.001, momentum = 0.9)
# fine_optimizer = optim.SGD([{'params': fine_model.conv1.parameters(), 'lr': 0.001},{'params': fine_model.conv2.parameters(), 'lr': 0.01},{'params': fine_model.conv3.parameters(), 'lr': 0.001}], lr = 0.001, momentum = 0.9)

# Changed values
# coarse_optimizer = optim.SGD([{'params': coarse_model.conv1.parameters(), 'lr': 0.01},{'params': coarse_model.conv2.parameters(), 'lr': 0.01},{'params': coarse_model.conv3.parameters(), 'lr': 0.01},{'params': coarse_model.conv4.parameters(), 'lr': 0.01},{'params': coarse_model.conv5.parameters(), 'lr': 0.01},{'params': coarse_model.fc1.parameters(), 'lr': 0.1},{'params': coarse_model.fc2.parameters(), 'lr': 0.1}], lr = 0.01, momentum = 0.9)
# fine_optimizer = optim.SGD(fine_model.parameters(), lr=args.lr, momentum=args.momentum)
# fine modified but default fine work more.
#fine_optimizer = optim.SGD([{'params': coarse_model.conv1.parameters(), 'lr': 0.01},{'params': coarse_model.conv2.parameters(), 'lr': 0.1},{'params': coarse_model.conv3.parameters(), 'lr': 0.01}], lr = 0.01, momentum = 0.9)

# default SGD optimiser - don't work
# # coarse_optimizer = optim.SGD(coarse_model.parameters(), lr=args.lr, momentum=args.momentum)
# fine_optimizer = optim.SGD(fine_model.parameters(), lr=args.lr, momentum=args.momentum)

# coarse_optimizer = optim.Adadelta(coarse_model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
# fine_optimizer = optim.Adadelta(fine_model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

# coarse_optimizer = optim.Adagrad(coarse_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
# fine_optimizer = optim.Adagrad(fine_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)

coarse_optimizer = optim.Adam(coarse_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
fine_optimizer = optim.Adam(fine_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# coarse_optimizer = optim.Adamax(coarse_model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# fine_optimizer = optim.Adamax(fine_model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# coarse_optimizer = optim.ASGD(coarse_model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
# fine_optimizer = optim.ASGD(fine_model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)


dtype=torch.cuda.FloatTensor
logger = Logger('./logs/' + args.model_folder)

def custom_loss_function(output, target):
    # di = output - target
    di = target - output
    n = (output_height * output_width)
    di2 = torch.pow(di, 2)
    first_term = torch.sum(di2,(1,2,3))/n
    second_term = 0.5*torch.pow(torch.sum(di,(1,2,3)), 2)/ (n**2)
    loss = first_term - second_term
    return loss.mean()

def scale_invariant(output, target):
    # di = output - target
    di = target - output
    n = (output_height * output_width)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2,(1,2,3))/n
    second_term = torch.pow(torch.sum(di,(1,2,3)), 2)/ (n**2)
    loss = fisrt_term - second_term
    return loss.mean()

# def custom_loss_function(output, target):
#     diff = target - output
#     alpha = torch.sum(diff, (1,2,3))/(output_height * output_width)
#     loss_val = 0
#     for i in range(alpha.shape[0]):
#        loss_val += torch.sum(torch.pow(((output[i] - target[i]) - alpha[i]), 2))/(2 * output_height * output_width)
#     loss_val = loss_val/output.shape[0] 
#     return loss_val

# All Error Function
def threeshold_percentage(output, target, threeshold_val):
    d1 = torch.exp(output)/torch.exp(target)
    d2 = torch.exp(target)/torch.exp(output)
    # d1 = output/target
    # d2 = target/output
    max_d1_d2 = torch.max(d1,d2)
    zero = torch.zeros(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    one = torch.ones(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    bit_mat = torch.where(max_d1_d2.cpu() < threeshold_val, one, zero)
    count_mat = torch.sum(bit_mat, (1,2,3))
    threeshold_mat = count_mat/(output.shape[2] * output.shape[3])
    return threeshold_mat.mean()

def rmse_linear(output, target):
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    # actual_output = output
    # actual_target = target
    diff = actual_output - actual_target
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (1,2,3))/(output.shape[2] * output.shape[3])
    rmse = torch.sqrt(mse)
    return rmse.mean()

def rmse_log(output, target):
    diff = output - target
    # diff = torch.log(output) - torch.log(target)
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (1,2,3))/(output.shape[2] * output.shape[3])
    rmse = torch.sqrt(mse)
    return mse.mean()

def abs_relative_difference(output, target):
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    # actual_output = output
    # actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target)/actual_target
    abs_relative_diff = torch.sum(abs_relative_diff, (1,2,3))/(output.shape[2] * output.shape[3])
    return abs_relative_diff.mean()

def squared_relative_difference(output, target):
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    # actual_output = output
    # actual_target = target
    square_relative_diff = torch.pow(torch.abs(actual_output - actual_target), 2)/actual_target
    square_relative_diff = torch.sum(square_relative_diff, (1,2,3))/(output.shape[2] * output.shape[3])
    return square_relative_diff.mean()    

def train_coarse(epoch):
    coarse_model.train()
    train_coarse_loss = 0
#     for batch_idx, data in enumerate(train_loader):
    for batch_idx, data in enumerate(train_loader_coarse):
        # variable
        rgb, depth = torch.tensor(data['image'].cuda(), requires_grad = True), torch.tensor(data['depth'].cuda(), requires_grad = True)
        # rgb = data['image'].cuda().clone().detach().requires_grad_(True)
        # depth = data['depth'].cuda().clone().detach().requires_grad_(True)
        coarse_optimizer.zero_grad()
        output = coarse_model(rgb.type(dtype))
        loss = custom_loss_function(output, depth)
        loss.backward()
        coarse_optimizer.step()
        train_coarse_loss += loss.item()
    train_coarse_loss /= (batch_idx + 1)
    # save loss
    with torch.no_grad():
        # print(train_coarse_loss)
        print(batch_idx+1)
        coarse_train_loss_hist.append(train_coarse_loss)
    print('Epoch: {} Training set(Coarse) average loss: {:.4f}'.format(epoch, train_coarse_loss))
    return train_coarse_loss
        # if batch_idx % args.log_interval == 0:
        #     training_tag = "coarse training loss epoch:" + str(epoch)
        #     logger.scalar_summary(training_tag, loss.item(), batch_idx)

        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(rgb), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

def train_fine(epoch):
    coarse_model.eval()
    fine_model.train()
    train_fine_loss = 0
#     for batch_idx, data in enumerate(train_loader):
    for batch_idx, data in enumerate(train_loader_fine):
        # variable
        rgb, depth = torch.tensor(data['image'].cuda(), requires_grad = True), torch.tensor(data['depth'].cuda(), requires_grad = True)
        # rgb = data['image'].cuda().clone().detach().requires_grad_(True)
        # depth = data['depth'].cuda().clone().detach().requires_grad_(True)
        fine_optimizer.zero_grad()
        coarse_output = coarse_model(rgb.type(dtype))   # it should print last epoch error since coarse is fixed.
        output = fine_model(rgb.type(dtype), coarse_output.type(dtype))
        loss = custom_loss_function(output, depth)
        loss.backward()
        fine_optimizer.step()
        train_fine_loss += loss.item()
    train_fine_loss /= (batch_idx + 1)
    # save loss
    fine_train_loss_hist.append(train_fine_loss)
    print('Epoch: {} Training set(Fine) average loss: {:.4f}'.format(epoch, train_fine_loss))
    return train_fine_loss
        # if batch_idx % args.log_interval == 0:
        #     training_tag = "fine training loss epoch:" + str(epoch)
        #     logger.scalar_summary(training_tag, loss.item(), batch_idx)

        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(rgb), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

def coarse_validation(epoch, training_loss):
    global prev_coarse_val_accum
    coarse_model.eval()
    coarse_validation_loss = 0
    scale_invariant_loss = 0
    delta1_accuracy = 0
    delta2_accuracy = 0
    delta3_accuracy = 0
    rmse_linear_loss = 0
    rmse_log_loss = 0
    abs_relative_difference_loss = 0
    squared_relative_difference_loss = 0
    
    # save img
    out_batch = [0]
    epoch_out = [1, 5, 10]
    img_count = 0
#     for batch_idx, data in enumerate(val_loader):
    for batch_idx, data in enumerate(val_loader_coarse):
        rgb, depth = torch.tensor(data['image'].cuda(), requires_grad = False), torch.tensor(data['depth'].cuda(), requires_grad = False)
        # rgb = data['image'].cuda().clone().detach().requires_grad_(False)
        # depth = data['depth'].cuda().clone().detach().requires_grad_(False)
        # variable modified by Yizhi 
        # rgb_orig, depth_orig = torch.tensor(data['original_image'].cuda(), requires_grad = False), torch.tensor(data['original_depth'].cuda(), requires_grad = False)
        coarse_output = coarse_model(rgb.type(dtype))
        '''
        # save img modified by Yizhi 
        if epoch in epoch_out and batch_idx in out_batch and img_count == 0:
        # if epoch == args.epochs and batch_idx in out_batch and img_count == 0:
            rgb_local = rgb_orig[0].cpu().numpy()
            depth_local = depth_orig[0].cpu().numpy()# [:, :, 0]
#             depth_local = np.exp(depth_local)
            output_local = coarse_output[0].cpu().detach().numpy()[0]
            output_local = np.exp(output_local)
            # print('rgb=\n', rgb_local.shape, rgb_local.transpose(1, 2, 0).shape)
            print('label_depth=\n', depth_local.shape, depth_local)
            print('out=\n', output_local.shape, output_local)
            img_count = 1
            if epoch == 1:
                rgb_img = Image.fromarray(rgb_local, 'RGB')
                rgb_img.save(img_out_dir + "_rgb.png")
                depth_img = Image.fromarray(depth_local) #.convert('RGB')
                depth_img.save(img_out_dir + "_label.png")
#             output_img = Image.fromarray(output_local).convert('RGB')
#             output_img.save(img_out_dir + '/epoch' + str(epoch) + "_course_out.png")
        '''
        with torch.no_grad():
            coarse_validation_loss += custom_loss_function(coarse_output, depth).item()
            # all error functions
            scale_invariant_loss += scale_invariant(coarse_output, depth)
            delta1_accuracy += threeshold_percentage(coarse_output, depth, 1.25)
            delta2_accuracy += threeshold_percentage(coarse_output, depth, 1.25*1.25)
            delta3_accuracy += threeshold_percentage(coarse_output, depth, 1.25*1.25*1.25)
            rmse_linear_loss += rmse_linear(coarse_output, depth)
            rmse_log_loss += rmse_log(coarse_output, depth)
            abs_relative_difference_loss += abs_relative_difference(coarse_output, depth)
            squared_relative_difference_loss += squared_relative_difference(coarse_output, depth)
    # need to fix logger
    # logger.scalar_summary("coarse validation loss", coarse_validation_loss, epoch)
    # save loss and acc modified by Yizhi (0511)    
    with torch.no_grad():
        coarse_validation_loss /= (batch_idx + 1)
        delta1_accuracy /= (batch_idx + 1)
        delta2_accuracy /= (batch_idx + 1)
        delta3_accuracy /= (batch_idx + 1)
        rmse_linear_loss /= (batch_idx + 1)
        rmse_log_loss /= (batch_idx + 1)
        abs_relative_difference_loss /= (batch_idx + 1)
        squared_relative_difference_loss /= (batch_idx + 1)
        
        coarse_val_loss_hist.append(coarse_validation_loss)
        coarse_val_acc_hist.append(delta3_accuracy)
        coarse_val_rmse_lin_hist.append(rmse_linear_loss)
        coarse_val_rmse_log_hist.append(rmse_log_loss)
        
        # zz modified 0511
        if epoch == 1:
            prev_coarse_val_accum = coarse_validation_loss
        else:
            prev_coarse_val_accum = prev_coarse_val_accum * 0.4 + coarse_validation_loss * 0.6
        coarse_val_accum_err.append(prev_coarse_val_accum)
        
    print('\nValidation set: Average loss(Coarse): {:.4f} \n'.format(coarse_validation_loss))
    print('Epoch: {}    {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}'.format(epoch, training_loss, 
        coarse_validation_loss, delta1_accuracy, delta2_accuracy, delta3_accuracy, rmse_linear_loss, rmse_log_loss, 
        abs_relative_difference_loss, squared_relative_difference_loss))

def fine_validation(epoch, training_loss):
    global prev_fine_val_accum
    fine_model.eval()
    fine_validation_loss = 0
    scale_invariant_loss = 0
    delta1_accuracy = 0
    delta2_accuracy = 0
    delta3_accuracy = 0
    rmse_linear_loss = 0
    rmse_log_loss = 0
    abs_relative_difference_loss = 0
    squared_relative_difference_loss = 0
    
    # save img
    out_batch = [0]
    epoch_out = [1, 5, 10]
    img_count = 0
#     for batch_idx,data in enumerate(val_loader):
    for batch_idx,data in enumerate(val_loader_fine):
        # variable
        rgb, depth = torch.tensor(data['image'].cuda(), requires_grad = False), torch.tensor(data['depth'].cuda(), requires_grad = False)
        # rgb = data['image'].cuda().clone().detach().requires_grad_(False)
        # depth = data['depth'].cuda().clone().detach().requires_grad_(False)
        coarse_output = coarse_model(rgb.type(dtype))
        fine_output = fine_model(rgb.type(dtype), coarse_output.type(dtype))
        '''
        # save img modified by Yizhi
        if epoch in epoch_out and batch_idx in out_batch and img_count == 0:
        # if epoch == args.epochs and batch_idx in out_batch and img_count == 0:
            output_local = fine_output[0].cpu().detach().numpy()[0]
            print('out=\n', output_local.shape)
            img_count = 1
            output_img = Image.fromarray(output_local).convert('RGB')
            otuput_img.save(img_out_dir + '/epoch' + str(epoch) + "_fine_out.png")
        '''
        with torch.no_grad():
            fine_validation_loss += custom_loss_function(fine_output, depth).item()
            # all error functions
            scale_invariant_loss += scale_invariant(fine_output, depth)
            delta1_accuracy += threeshold_percentage(fine_output, depth, 1.25)
            delta2_accuracy += threeshold_percentage(fine_output, depth, 1.25*1.25)
            delta3_accuracy += threeshold_percentage(fine_output, depth, 1.25*1.25*1.25)
            rmse_linear_loss += rmse_linear(fine_output, depth)
            rmse_log_loss += rmse_log(fine_output, depth)
            abs_relative_difference_loss += abs_relative_difference(fine_output, depth)
            squared_relative_difference_loss += squared_relative_difference(fine_output, depth)
    with torch.no_grad():
        fine_validation_loss /= (batch_idx + 1)
        scale_invariant_loss /= (batch_idx + 1)
        delta1_accuracy /= (batch_idx + 1)
        delta2_accuracy /= (batch_idx + 1)
        delta3_accuracy /= (batch_idx + 1)
        rmse_linear_loss /= (batch_idx + 1)
        rmse_log_loss /= (batch_idx + 1)
        abs_relative_difference_loss /= (batch_idx + 1)
        squared_relative_difference_loss /= (batch_idx + 1)
        # need to fix logger
        # logger.scalar_summary("fine validation loss", fine_validation_loss, epoch)
        # save loss and acc modified by Yizhi (0511)
        
        fine_val_loss_hist.append(fine_validation_loss)
        fine_val_acc_hist.append(delta3_accuracy)
        fine_val_rmse_lin_hist.append(rmse_linear_loss)
        fine_val_rmse_log_hist.append(rmse_log_loss)
        
        if epoch == 1:
            prev_fine_val_accum = fine_validation_loss
        else:
            prev_fine_val_accum = prev_fine_val_accum * 0.4 + fine_validation_loss * 0.6
        fine_val_accum_err.append(prev_fine_val_accum)
        
    print('\nValidation set: Average loss(Fine): {:.4f} \n'.format(fine_validation_loss))
    print('Epoch: {}    {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}'.format(epoch, training_loss, 
        fine_validation_loss, delta1_accuracy, delta2_accuracy, delta3_accuracy, rmse_linear_loss, rmse_log_loss, 
        abs_relative_difference_loss, squared_relative_difference_loss))

folder_name = "models/" + args.model_folder
if not os.path.exists(folder_name): os.mkdir(folder_name)
    
# print("Transformed Length:", len(transformed_train_loader))
print("********* Training the Coarse Model **************")
print("Epochs:     Train_loss  Val_loss    Delta_1     Delta_2     Delta_3    rmse_lin    rmse_log    abs_rel.  square_relative")
print("Paper Val:                          (0.618)     (0.891)     (0.969)     (0.871)     (0.283)     (0.228)     (0.223)")

for epoch in range(1, args.epochs + 1):
    # print("********* Training the Coarse Model **************")
    training_loss = train_coarse(epoch)
    coarse_validation(epoch, training_loss)
    model_file = folder_name + "/" + 'coarse_model_' + str(epoch) + '.pth'
    if(epoch%10 == 0):
        torch.save(coarse_model.state_dict(), model_file)
    # empty cache modified by Yizhi (0511)
    torch.cuda.empty_cache()

coarse_model.eval() # stoping the coarse model to train.

np.savetxt(txt_out_dir+'/coarse_train_loss', coarse_train_loss_hist, fmt='%f')
np.savetxt(txt_out_dir+'/coarse_val_loss', coarse_val_loss_hist, fmt='%f')
np.savetxt(txt_out_dir+'/coarse_val_acc', coarse_val_acc_hist, fmt='%f')
np.savetxt(txt_out_dir+'/coarse_val_rmse_lin', coarse_val_rmse_lin_hist, fmt='%f')
np.savetxt(txt_out_dir+'/coarse_val_rmse_log', coarse_val_rmse_log_hist, fmt='%f')
np.savetxt(txt_out_dir+'/coarse_val_avg_loss', coarse_val_accum_err, fmt='%f')


print("********* Training the Fine Model ****************")
print("Epochs:     Train_loss  Val_loss    Delta_1     Delta_2     Delta_3    rmse_lin    rmse_log    abs_rel.  square_relative")
print("Paper Val:                          (0.611)     (0.887)     (0.971)     (0.907)     (0.285)     (0.215)     (0.212)")
for epoch in range(1, args.epochs + 1):
    # print("********* Training the Fine Model ****************")
    training_loss = train_fine(epoch)
    fine_validation(epoch, training_loss)
    model_file = folder_name + "/" + 'fine_model_' + str(epoch) + '.pth'
    if(epoch%5 == 0):
        torch.save(fine_model.state_dict(), model_file)
    # empty cache modified by Yizhi (0511)
    torch.cuda.empty_cache()


np.savetxt(txt_out_dir+'/fine_train_loss', fine_train_loss_hist, fmt='%f')
np.savetxt(txt_out_dir+'/fine_val_loss', fine_val_loss_hist, fmt='%f')
np.savetxt(txt_out_dir+'/fine_val_acc', fine_val_acc_hist, fmt='%f')
np.savetxt(txt_out_dir+'/fine_val_rmse_lin', fine_val_rmse_lin_hist, fmt='%f')
np.savetxt(txt_out_dir+'/fine_val_rmse_log', fine_val_rmse_log_hist, fmt='%f')
np.savetxt(txt_out_dir+'/fine_val_avg_loss', fine_val_accum_err, fmt='%f')
