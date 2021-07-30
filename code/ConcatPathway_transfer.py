# get_ipython().run_line_magic('matplotlib', 'inline')
#!/usr/bin/env python
# coding: utf-8

'''
sample command: python try_ConcatPathway.py -c 1 --var 1 --wr1 0.001 --wr2 0.001 --wkl 0.0005 --res 1 --ep 1000 -i 1 -k 0 -r 1 --data_dir ../data --spv 1 
Individual training for BioTac data (full/partial data)
if -r=1, train with full data
if -r=2, train with half data
loss = classification loss + recon loss 
'''

# Import
import os,sys
import pickle
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from vrae.ConcatPathway import *
from preprocess_data import get_TrainValTestLoader, get_TrainValTestDataset, get_TrainValTestData, property_label

import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix
import copy

def main(args):
    # Set hyper params
    args_data_dir = args.data_dir
    kfold_number = args.kfold
    data_reduction_ratio = args.reduction
    spv = args.spv
    var = int(args.var)
    rep = args.rep
    residual=args.res
    n_epochs = args.ep
    data_folder = 'c20/'


    # param for loading data
    shuffle = True # set to False for partial training
    z_norm_preprocessing = False
    sequence_length = 400 # total seq_len for loading data, cut into seg & different freq are processed within the model
    number_of_features = 19
    x_freq = 100
    # param for receptor setting
    receptor_input_size = 18
    receptor_freq = [20, 10]
    receptor_hidden_size = [100, 200]
    receptor_block = "GRU" #"LSTM"
    num_receptor = len(receptor_freq)
    # param for mid neuron setting
    mid_hidden_size = 90
    mid_block = "LSTM" # "LMU" # "LSTM"
    mid_lmu_order = 3
    latent_length = 40
    
    # general model setting
    header = 'MLP'
    dropout = 0
    bidirectional = False
    # training setting
    batch_size = 32
    learning_rate = 0.0005 # 0.0005
    cuda = True # options: True, False

    use_stiffness = args.use_stiffness
    use_roughness = args.use_roughness
    p_threshold = args.p_threshold
    w_p = args.w_p

    if data_folder == 'c20icub/':
        header = 'CNN'
        number_of_features = 60
        sequence_length = 75
        x_freq = 50


    # loss weightage
    w_c = 1
    w_r1 = args.wr1
    w_r2 = args.wr2
    w_kl = args.wkl

    np.random.seed(1)
    torch.manual_seed(1)

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.cuda))
    else:
        device = torch.device('cpu')

    if args.reduction != 1:
        print("load {} kfold number, reduce data to {} folds, put to device: {}".format(args.kfold, args.reduction, device))
    else:
        print("load {} kfold number, train with full data, put to devide: {}".format(args.kfold, device))

    # Load data
    logDir = 'models_and_stats/'
    if_plot = False
    test_only = False


    # Load new data
    if z_norm_preprocessing:
        prefix = "_znorm_"
    else:
        prefix = ""

    
    num_class = int(data_folder[1:3])
    print("use data folder {}, num_class {}".format(data_folder, num_class))
    dataset_dir = os.path.join(args_data_dir, data_folder)

    train_set, val_set, test_set = get_TrainValTestDataset(dataset_dir, k=kfold_number, prefix=prefix, seq_len=sequence_length)
    train_loader, val_loader, test_loader = get_TrainValTestLoader(dataset_dir, k=kfold_number, batch_size=batch_size,shuffle=shuffle, prefix=prefix,seq_len=sequence_length)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = get_TrainValTestData(dataset_dir, k=kfold_number, prefix=prefix,seq_len=sequence_length)

    # set color_list for future plots
    color_list = ['b','g','r','c','m','y','dimgray', 'darkorange', 'indigo', 'teal', 'coral']

    # Initialize models
    eheader_norm = None # "BN"
    receptor_norm = None # "BN"
    mid_norm = None # "BN"
    model_name = "BT19_{}_ConcatPathway_stiffness{}_roughness{}_p_thre_{}_rs_{}_ll{}_ihs{}_ohs{}_rf{}_res{}_rec{}_mid{}o{}_ep{}_lr{}_wp_{}_wc{}_wr1{}_wr2{}_wkl{}_var{}_rep{}_k{}".format(str(use_stiffness), str(use_roughness), str(p_threshold), str(args.rand_seed), data_folder[:-1], latent_length, receptor_hidden_size, mid_hidden_size, str(receptor_freq), residual, receptor_block, mid_block, mid_lmu_order, n_epochs, learning_rate, w_p, w_c, w_r1, w_r2, w_kl, var, rep, str(kfold_number))

    model = ConcatPathway(number_of_features=number_of_features,
                        num_class=num_class,
                        x_freq=x_freq,
                        header=header,
                        receptor_input_size=receptor_input_size,
                        receptor_freq=receptor_freq,
                        receptor_hidden_size=receptor_hidden_size,
                        dropout=dropout,
                        bidirectional=bidirectional,
                        receptor_block=receptor_block,
                        mid_hidden_size=mid_hidden_size,
                        mid_block=mid_block,
                        mid_lmu_order=mid_lmu_order,
                        latent_length=latent_length,
                        residual=residual,
                        device = device,
                        var=var,
                        eheader_norm=eheader_norm,
                        receptor_norm=receptor_norm,
                        mid_norm=mid_norm)
    model.to(device)


    if test_only:
        num_sample = [2,5]
        plot_acc_loss(logDir, model_name, test_loader, kfold_number, if_plot)
        evaluate_concat_decoder(logDir, model_name, test_loader, kfold_number, if_plot, num_sample)


    # Initialize training settings
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cl_loss_fn = nn.CrossEntropyLoss()
    recon_loss_fn = nn.MSELoss()

    training_start=datetime.now()
    # create empty lists to fill stats later
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_loss = []
    epoch_val_acc = []
    epoch_train_cl_loss = []
    epoch_train_r1_loss = []
    epoch_train_r2_loss = []
    epoch_train_kl_loss = []
    epoch_val_cl_loss = []
    epoch_val_r1_loss = []
    epoch_val_r2_loss = []
    epoch_val_kl_loss = []
    epoch_train_stiffness_loss = []
    epoch_train_stiffness_acc = []
    epoch_val_stiffness_loss = []
    epoch_val_stiffness_acc = []
    epoch_train_roughness_loss = []
    epoch_train_roughness_acc = []
    epoch_val_roughness_loss = []
    epoch_val_roughness_acc = []
    max_val_acc = 0

    for epoch in range(n_epochs):
        
        # TRAIN
        model.train()
        correct = 0
        train_loss = 0
        train_num = 0
        
        train_cl_loss = 0
        train_r1_loss = 0
        train_r2_loss = 0
        train_kl_loss = 0

        correct_stiffness = 0
        train_stiffness_loss = 0
        correct_roughness = 0
        train_roughness_loss = 0
        
        
        for i, (x, y) in enumerate(train_loader):
            
            x, y = x.to(device), y.long().to(device)
            # reduce data by data_reduction_ratio times
            if i % data_reduction_ratio == 0:
                train_num += x.size(0)
                optimizer.zero_grad()
                inner_h_n, dec_r_out, enc_h_n, dec_input, latent, output = model(x)

                stiff_y, rough_y = property_label(y)
                stiff_y, rough_y = stiff_y.type(torch.FloatTensor), rough_y.type(torch.FloatTensor)
                stiff_y, rough_y = stiff_y.to(device), rough_y.to(device)

                cl_loss_stiffness = recon_loss_fn(latent[:, :, 0], stiff_y)
                cl_loss_roughness = recon_loss_fn(latent[:, :, 1], rough_y)

                # compute classification acc
                correct_stiffness += torch.sum((torch.abs(latent[:, :, 0] - stiff_y) < p_threshold))
                correct_roughness += torch.sum((torch.abs(latent[:, :, 1] - rough_y) < p_threshold))

                # construct loss function
                cl_loss = cl_loss_fn(output, y)
                r1_loss = recon_loss_fn(enc_h_n, dec_input)
                r2_loss = recon_loss_fn(inner_h_n, dec_r_out)
                loss = w_r1 *r1_loss + w_r2*r2_loss
                if spv:
                    loss += w_c*cl_loss
                if var:
                    latent_mean, latent_logvar = model.mid2latent.mean, model.mid2latent.logvar
                    kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
                    loss += w_kl*kl_loss
                else:
                    kl_loss = torch.tensor(0)
                if use_stiffness:
                    loss += w_p * cl_loss_stiffness
                if use_roughness:
                    loss +=  w_p * cl_loss_roughness
                
                # compute classification acc
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
                # accumulator
                train_loss += loss.item()
                train_cl_loss += w_c*(cl_loss.item())
                train_r1_loss += w_r1*r1_loss.item()
                train_r2_loss += w_r2*r2_loss.item()
                train_kl_loss += w_kl*kl_loss.item()
                train_stiffness_loss += cl_loss_stiffness.item()
                train_roughness_loss += cl_loss_roughness.item()

                if i == 0: # and epoch%50 == 0:
                    print('cl_loss:', cl_loss, 'cl_loss_stiffness:', cl_loss_stiffness, 'cl_loss_roughness:', cl_loss_roughness, 'recon_loss:', r1_loss, r1_loss)
                  
                
                loss.backward()
    #             figname = logDir + model_name + "grad_flow_plot" +".png"
    #             if i == 0 and epoch%50 == 0:
    #                 print("grad flow for epoch {}".format(epoch))
    #                 plot_grad_flow(model.named_parameters(), figname, if_plot)
                optimizer.step()
        
        if epoch < 10 or epoch%200 == 0:
            print("train last batch {} of {}: cl_loss {:.3f}, r1_loss {:.3f}, r2_loss {:.3f}, kl_loss {:.3f}".format
                (i,len(train_loader), cl_loss, r1_loss, r2_loss, kl_loss))

        # fill stats
        train_accuracy = correct/ train_num
        epoch_train_loss.append(train_loss/ train_num )
        epoch_train_acc.append(train_accuracy) 
        
        epoch_train_cl_loss.append(train_cl_loss/ train_num )
        epoch_train_r1_loss.append(train_r1_loss/ train_num )
        epoch_train_r2_loss.append(train_r2_loss/ train_num )
        epoch_train_kl_loss.append(train_kl_loss/ train_num )

        train_stiffness_accuracy = correct_stiffness.item() / train_num
        train_stiffness_loss /= train_num
        epoch_train_stiffness_loss.append(train_stiffness_loss) 
        epoch_train_stiffness_acc.append(train_stiffness_accuracy)
        train_roughness_accuracy = correct_roughness.item() / train_num 
        train_roughness_loss /= train_num
        epoch_train_roughness_loss.append(train_roughness_loss) 
        epoch_train_roughness_acc.append(train_roughness_accuracy) 
        
        
        # VALIDATION
        model.eval()
        correct = 0
        val_loss = 0
        val_num = 0
        
        val_cl_loss = 0
        val_r1_loss = 0
        val_r2_loss = 0
        val_kl_loss = 0

        correct_stiffness = 0
        val_stiffness_loss = 0
        correct_roughness = 0
        val_roughness_loss = 0 
        
        for i, (x, y) in enumerate(val_loader):

            x, y = x.to(device), y.long().to(device)
            val_num += x.size(0)
            
            inner_h_n, dec_r_out, enc_h_n, dec_input, latent, output = model(x)

            stiff_y, rough_y = property_label(y)
            stiff_y, rough_y = stiff_y.to(device), rough_y.to(device)
            cl_loss = cl_loss_fn(output, y)
            cl_loss_stiffness = recon_loss_fn(latent[:, :, 0], stiff_y)
            cl_loss_roughness = recon_loss_fn(latent[:, :, 1], rough_y)
            # compute classification acc
            correct_stiffness += torch.sum((torch.abs(latent[:, :, 0] - stiff_y) < p_threshold))
            correct_roughness += torch.sum((torch.abs(latent[:, :, 1] - rough_y) < p_threshold))

            # construct loss function
            cl_loss = cl_loss_fn(output, y)
            r1_loss = recon_loss_fn(enc_h_n, dec_input)
            r2_loss = recon_loss_fn(inner_h_n, dec_r_out)
            loss = w_r1 *r1_loss + w_r2*r2_loss
            if spv:
                loss += w_c*cl_loss
            if var:
                latent_mean, latent_logvar = model.mid2latent.mean, model.mid2latent.logvar
                kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
                loss += w_kl*kl_loss
            else:
                kl_loss = torch.tensor(0)
            
            if use_stiffness:
                loss += w_p * cl_loss_stiffness
            if use_roughness:
                loss +=  w_p * cl_loss_roughness
            
            # compute classification acc
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
            # accumulator
            val_loss += loss.item()
            val_cl_loss += w_c*(cl_loss.item())
            val_r1_loss += w_r1*r1_loss.item()
            val_r2_loss += w_r2*r2_loss.item()
            val_kl_loss += w_kl*kl_loss.item()
            val_stiffness_loss += cl_loss_stiffness.item()
            val_roughness_loss += cl_loss_roughness.item()

        # fill stats
        val_accuracy = correct / val_num
        epoch_val_loss.append(val_loss/ val_num)  # only save the last batch
        epoch_val_acc.append(val_accuracy)
        epoch_val_cl_loss.append(train_cl_loss/ train_num )
        epoch_val_r1_loss.append(train_r1_loss/ train_num )
        epoch_val_r2_loss.append(train_r2_loss/ train_num )
        epoch_val_kl_loss.append(train_kl_loss/ train_num )
        val_stiffness_accuracy = correct_stiffness.item() / val_num 
        val_stiffness_loss /= val_num        
        epoch_val_stiffness_loss.append(val_stiffness_loss) 
        epoch_val_stiffness_acc.append(val_stiffness_accuracy) 
        val_roughness_accuracy = correct_roughness.item() / val_num 
        val_roughness_loss /= val_num
        epoch_val_roughness_loss.append(val_roughness_loss) 
        epoch_val_roughness_acc.append(val_roughness_accuracy)       
        
        # if epoch < 10 or epoch%200 == 0:
        print("train_num {}, val_num {}".format(train_num, val_num))
        print('Epoch: {} Loss: train {:.3f}, valid {:.3f}. Accuracy: train: {:.3f}, valid {:.3f}. Stiffness Accuracy: train: {:.3f}, valid {:.3f}. Roughness Accuracy: train: {:.3f}, valid {:.3f}'.format(epoch, train_loss, val_loss, train_accuracy, val_accuracy, train_stiffness_accuracy, val_stiffness_accuracy, train_roughness_accuracy, val_roughness_accuracy))
        
        # choose model
        if max_val_acc <= val_accuracy:
            model_dir = logDir + model_name + str(epoch) + '.pt'
            print('Saving model at {} epoch to {}'.format(epoch, model_dir))
            max_val_acc = val_accuracy
            max_val_epoch = epoch
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_dir)
        if epoch == n_epochs-1:
            model_dir = logDir + model_name + str(epoch) + '.pt'
            print('Saving model at {} epoch to {}'.format(epoch, model_dir))
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_dir)

            
    training_end =  datetime.now()
    training_time = training_end -training_start 
    print("training takes time {}".format(training_time))

    model.is_fitted = True
    model.eval()


    # TEST
    correct = 0
    test_num = 0
    correct_stiffness = 0
    correct_roughness = 0
    for i, (x, y) in enumerate(test_loader):

        x, y = x.to(device), y.long().to(device)
        test_num += x.size(0)
        inner_h_n, dec_r_out, enc_h_n, dec_input, latent, output = model(x)
        
        # compute classification acc
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
        stiff_y, rough_y = property_label(y)
        stiff_y, rough_y = stiff_y.to(device), rough_y.to(device)
        correct_stiffness += torch.sum((torch.abs(latent[:, :, 0] - stiff_y) < p_threshold))
        correct_roughness += torch.sum((torch.abs(latent[:, :, 1] - rough_y) < p_threshold))
        
    test_acc = correct / test_num #len(test_loader.dataset)
    test_stiffness_acc = correct_stiffness.item() / test_num
    test_roughness_acc = correct_roughness.item() / test_num
    print('last epoch Test accuracy for', str(kfold_number), ' fold : ', test_acc, test_stiffness_acc, test_roughness_acc)

    # Save stats
    results_dict = {"epoch_train_loss": epoch_train_loss,
                    "epoch_train_acc": epoch_train_acc,
                    "epoch_val_loss": epoch_val_loss,
                    "epoch_val_acc": epoch_val_acc,
                    "epoch_train_cl_loss": epoch_train_cl_loss,
                    "epoch_train_r1_loss": epoch_train_r1_loss,
                    "epoch_train_r2_loss": epoch_train_r2_loss,
                    "epoch_train_kl_loss": epoch_train_kl_loss,
                    "epoch_val_cl_loss": epoch_val_cl_loss,
                    "epoch_val_r1_loss": epoch_val_r1_loss,
                    "epoch_val_r2_loss": epoch_val_r2_loss,
                    "epoch_val_kl_loss": epoch_val_kl_loss,
                    "test_acc": test_acc,
                    "training_time": training_time,
                    "epoch_train_stiffness_loss": epoch_train_stiffness_loss,
                    "epoch_train_stiffness_acc": epoch_train_stiffness_acc,
                    "epoch_val_stiffness_loss": epoch_val_stiffness_loss,
                    "epoch_val_stiffness_acc": epoch_val_stiffness_acc,
                    "test_stiffness_acc": test_stiffness_acc,
                    "epoch_train_roughness_loss": epoch_train_roughness_loss,
                    "epoch_train_roughness_acc": epoch_train_roughness_acc,
                    "epoch_val_roughness_loss": epoch_val_roughness_loss,
                    "epoch_val_roughness_acc": epoch_val_roughness_acc,
                    "test_roughness_acc1": test_roughness_acc}
    

    # Evaluate the results
    # plot_acc_loss(logDir, model_name, test_loader, kfold_number, if_plot, rep)

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--rep", type=int, default=0, help='index of running repetition')
parser.add_argument('--data_dir', type=str, default="data/", help="DIR set in 'gh_download.sh' to store compiled_data")
parser.add_argument("-k", "--kfold", type=int, default=0, help="kfold_number for loading data")
parser.add_argument("-r", "--reduction", type=int, default=1, help="data reduction ratio for partial training")
parser.add_argument("-c", "--cuda", default=0, help="index of cuda gpu to use")
# add the following for testing AE/VAE spv/unspv
parser.add_argument("--spv", default=1, help="if use supervised learning manner")
parser.add_argument("--var", default=1, help="if use variational AE")
parser.add_argument("--wr1", type=float, default=0.001, help="w_r1")
parser.add_argument("--wr2", type=float, default=0.001, help="w_r2")
parser.add_argument("--wkl", type=float, default=0.0005, help="w_kl")
parser.add_argument("--res", type=float, default=1, help="residual")
parser.add_argument("--ep", type=int, default=100, help="epochs")
parser.add_argument("--data_folder", type=str, default="c20/", help="data_folder")
# parser.add_argument("--freq", type=int, default=20, help="receptor_freq")
parser.add_argument("--rand_seed", type=int, default=1, help='random seed')
parser.add_argument("--w_p", default=5.0, type=float, help="weight of property classification loss")
parser.add_argument("--use_stiffness", default=1, type=int, help="if to use stiffness")
parser.add_argument("--use_roughness", default=1, type=int, help="if to use roughness")
parser.add_argument("--latent_length", default=40, type=int, help="size for the texture latent space")
parser.add_argument("--sequence_length", default=400, type=int, help="length for the input time sequence")
parser.add_argument("--p_threshold", default=0.25, type=int, help="error threshold for property classification")

args = parser.parse_args()

# # dummy class to replace argparser, if running jupyter notebook
# class Args:
#     rep = 0
#     data_dir = "/home/ruihan/Desktop/HCNC/data/" #"../data/"
#     kfold = 0
#     cuda = '1'
#     spv = 1
#     var = 1
#     sleep = 1
#     reduction = 1
#     res = 0
#     ep = 100
# args = Args()


# for w1, w2, wkl in [(0.001, 0.0001, 0.0005), (0.001, 0.001, 0.0005)]:
#     args.wr1 = w1
#     args.wr2 = w2
#     args.wkl = wkl
#     print(args)
#     main_func(args)
args.ep = 5
args.w_p = 2
args.use_stiffness = 1
args.use_roughness = 1
# result = []
args.data_folder = 'c20/'  
args.kfold = 0
print(args)
test_acc, test_stiffness_acc, test_roughness_acc = main(args)
# result.append([test_acc, test_stiffness_acc, test_roughness_acc])
# print(result)

# args.ep = 1
# args.w_p = 0
# args.use_stiffness = 0
# args.use_roughness = 0
# result = []
# for data_folder in ['c20/', 'c20comb/', 'c20new/', 'c20icub/']:
#     args.data_folder = data_folder  
#     for k in range(5):
#         args.kfold = k
#         print(args)
#         test_acc, test_stiffness_acc, test_roughness_acc = main(args)
#         result.append([test_acc, test_stiffness_acc, test_roughness_acc])
#     print(result)
# print(result)
