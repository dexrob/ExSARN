#!/usr/bin/env python
# coding: utf-8

'''
sample command: python T4_BT19_ae.py -k 0 -c 0 -r 1 --data_dir /home/ruihan/data
Individual training for BioTac data (full/partial data)
if -r=1, train with full data
if -r=2, train with half data
loss = classification loss + recon loss 
'''

# Import
import os,sys
import math
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
import seaborn as sns

from vrae.vrae import VRAEC, VRAEC_v2
# c20_2: preprocess_data_copy
from preprocess_data import get_TrainValTestLoader, get_TrainValTestDataset, get_TrainValTestData, property_label
from vrae.visual import plot_grad_flow, plot_stats, tsne, multiple_tsne

def train(args):
    # Set hyper params
    args_data_dir = args.data_dir
    kfold_number = args.kfold
    data_reduction_ratio = args.reduction
    shuffle = True # set to False for partial training
    sequence_length = args.sequence_length
    number_of_features = 60

    hidden_size = args.h_s
    hidden_layer_depth = 1
    latent_length = args.latent_length
    batch_size = 32
    learning_rate = 0.001 # 0.0005
    n_epochs = 1
    dropout_rate = 0.2
    cuda = True # options: True, False
    header = 'CNN'
    dataset = args.dataset
    if dataset == 'c50':
        num_class_texture = 50
    else:
        num_class_texture = 20
    
    use_stiffness = args.use_stiffness
    num_class_stiffness = 3

    use_roughness = args.use_roughness
    num_class_roughness = 3

    p_threshold = args.p_threshold

    sample_percent = args.sample_percent

    # loss weightage
    w_r = args.w_r
    w_c = 1
    w_p = args.w_p

    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    # Load data
    # data_dir = os.path.join(args_data_dir, "compiled_data/")
    logDir = 'models_and_stats/'
    if_plot = False

    # RNN block
    block = "LSTM" # LSTM, GRU, phased_LSTM

    model_name = 'p_in_lat_stiffness{}_roughness{}_p_thre_{}_rs_{}_B_block_{}_data_{}_wrI_{}_wC_{}_wp_{}_hidden_{}_latent_{}_k_{}_p_{}'.format(str(use_stiffness), str(use_roughness), str(p_threshold), str(args.rand_seed), block, dataset, w_r, w_c, w_p, str(hidden_size), str(latent_length), str(kfold_number), str(sample_percent))

    if torch.cuda.is_available() and cuda:
        device = torch.device("cuda:{}".format(args.cuda))
    else:
        device = torch.device('cpu')

    if args.reduction != 1:
        print("load {} kfold number, reduce data to {} folds, put to device: {}".format(args.kfold, args.reduction, device))
    else:
        print("load {} kfold number, train with full data, put to device: {}".format(args.kfold, device))

    prefix = ""
    # if p is not None, k is the random seed for spliting train/val/test data
    dataset_dir = os.path.join(args_data_dir, dataset+"/") # TODO
    # train_set, val_set, test_set = get_TrainValTestDataset(dataset_dir, k=kfold_number, prefix=prefix, seq_len=sequence_length, p=sample_percent)
    train_loader, val_loader, test_loader = get_TrainValTestLoader(dataset_dir, k=kfold_number, batch_size=batch_size,shuffle=shuffle, prefix=prefix,seq_len=sequence_length, p=sample_percent, i=args.i)
    # X_train, X_val, X_test, Y_train, Y_val, Y_test = get_TrainValTestData(dataset_dir, k=kfold_number, prefix=prefix,seq_len=sequence_length, p=sample_percent)
    # Initialize models
    model = VRAEC(num_class=num_class_texture,
                block=block,
                sequence_length=sequence_length, # TODO
                number_of_features = number_of_features,
                hidden_size = hidden_size, 
                hidden_layer_depth = hidden_layer_depth,
                latent_length = latent_length,
                batch_size = batch_size,
                learning_rate = learning_rate,
                n_epochs = n_epochs,
                dropout_rate = dropout_rate,
                cuda = cuda,
                model_name=model_name,
                header=header,
                device = device)
    model.to(device)

    # Initialize training settings
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cl_loss_fn = nn.NLLLoss()
    recon_loss_fn = nn.MSELoss()

    # model.load_state_dict(torch.load('models_and_stats/model_phased_LSTM_B30.pt', map_location='cpu'))
    # saved_dicts = torch.load('models_and_stats/model_phased_LSTM_B.pt', map_location='cpu')
    # model.load_state_dict(saved_dicts['model_state_dict'])
    # optimizer.load_state_dict(saved_dicts['optimizer_state_dict'])

    training_start=datetime.now()
    # create empty lists to fill stats later
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_loss = []
    epoch_val_acc = []
    epoch_train_stiffness_loss = []
    epoch_train_stiffness_acc = []
    epoch_val_stiffness_loss = []
    epoch_val_stiffness_acc = []
    epoch_train_roughness_loss = []
    epoch_train_roughness_acc = []
    epoch_val_roughness_loss = []
    epoch_val_roughness_acc = []
    max_val_acc = 0
    max_val_epoch = 0
    if block == "phased_LSTM":
        time = torch.Tensor(range(sequence_length))
        times = time.repeat(batch_size, 1)

    for epoch in range(n_epochs):
        model.train()
        correct_texture = 0
        train_loss = 0
        train_num = 0
        correct_stiffness = 0
        train_stiffness_loss = 0
        correct_roughness = 0
        train_roughness_loss = 0
        
        for i, (XI,  y) in enumerate(train_loader):
            if model.header == 'CNN':
                x = XI
            else:
                x = XB
            x, y = x.to(device), y.long().to(device) # 32, 19, 400
            if x.size()[0] != batch_size:
                break
            
            # reduce data by data_reduction_ratio times
            if i % data_reduction_ratio == 0:
                train_num += x.size(0)
                optimizer.zero_grad()
                if block == "phased_LSTM":
                    x_decoded, latent, output = model(x, times)
                else:
                    x_decoded, latent, output_texture = model(x)

                # assert (output == 0).nonzero().size(0)==0, 'output contain zero, batch_num'+str(i)+' indices:'+str((output == 0).nonzero())
                if (output_texture == 0).nonzero().size(0) != 0:
                    print('batch_num'+str(i)+' indices:'+str((output_texture == 0).nonzero()))
                    cl_loss = cl_loss_fn(output_texture+1e-5, y) # avoid nan
                else:
                    cl_loss = cl_loss_fn(output_texture, y) 

                stiff_y, rough_y = property_label(y)
                stiff_y, rough_y = stiff_y.type(torch.FloatTensor), rough_y.type(torch.FloatTensor)
                stiff_y, rough_y = stiff_y.to(device), rough_y.to(device)

                cl_loss_stiffness = recon_loss_fn(latent[:, 0], stiff_y)
                cl_loss_roughness = recon_loss_fn(latent[:, 1], rough_y)

                # compute classification acc
                
                correct_stiffness += torch.sum((torch.abs(latent[:, 0] - stiff_y) < p_threshold))
                correct_roughness += torch.sum((torch.abs(latent[:, 1] - rough_y) < p_threshold))
                # if i==0:
                #     print((torch.abs(latent[:, 0] - stiff_y) < p_threshold))
                #     print(correct_stiffness)
                # cl_loss_stiffness = cl_loss_fn(output_stiffness, stiff_y)
                # cl_loss_roughness = cl_loss_fn(output_roughness, rough_y)

                recon_loss = recon_loss_fn(x_decoded, x)
                loss = w_c * cl_loss + w_r * recon_loss
                if use_stiffness:
                    loss += w_p * cl_loss_stiffness
                if use_roughness:
                    loss +=  w_p * cl_loss_roughness
                
                # # compute classification acc
                pred_texture = output_texture.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct_texture += pred_texture.eq(y.data.view_as(pred_texture)).long().cpu().sum().item()
                # pred_stiffness = output_stiffness.data.max(1, keepdim=True)[1]
                # correct_stiffness += pred_stiffness.eq(stiff_y.data.view_as(pred_stiffness)).long().cpu().sum().item()
                # pred_roughness = output_roughness.data.max(1, keepdim=True)[1]
                # correct_roughness += pred_roughness.eq(rough_y.data.view_as(pred_roughness)).long().cpu().sum().item()
                # accumulator
                train_loss += loss.item()
                train_stiffness_loss += cl_loss_stiffness.item()
                train_roughness_loss += cl_loss_roughness.item()
                start_bp = datetime.now()
                loss.backward()
                # if epoch == 0 and i == 0:
                    # tsne_data = x
                    # tsne_y = y, stiff_y, rough_y
                    # tsne(latent, y, model_name+'init')
                    # multiple_tsne(latent, y, stiff_y, rough_y, model_name+'init')

                if i == 0: # and epoch%50 == 0:
                    print('cl_loss:', cl_loss, 'cl_loss_stiffness:', cl_loss_stiffness, 'cl_loss_roughness:', cl_loss_roughness, 'recon_loss:', recon_loss)
                    
                optimizer.step()
                # print('1 batch bp time:', datetime.now()-start_bp)

        # if epoch == 0:
        #     print('first epoch training time:', datetime.now()-training_start)

        # fill stats
        train_accuracy = correct_texture / train_num 
        train_loss /= train_num
        epoch_train_loss.append(train_loss) 
        epoch_train_acc.append(train_accuracy) 
        train_stiffness_accuracy = correct_stiffness.item() / train_num
        # print('correct_stiffness / train_num', correct_stiffness, train_num) 
        train_stiffness_loss /= train_num
        epoch_train_stiffness_loss.append(train_stiffness_loss) 
        epoch_train_stiffness_acc.append(train_stiffness_accuracy)
        train_roughness_accuracy = correct_roughness.item() / train_num 
        train_roughness_loss /= train_num
        epoch_train_roughness_loss.append(train_roughness_loss) 
        epoch_train_roughness_acc.append(train_roughness_accuracy) 
        
        # VALIDATION
        model.eval()
        correct_texture = 0
        val_loss = 0
        val_num = 0
        correct_stiffness = 0
        val_stiffness_loss = 0
        correct_roughness = 0
        val_roughness_loss = 0    
        for i, (XI, y) in enumerate(val_loader):
            if model.header == 'CNN':
                x = XI
            else:
                x = XB
            # x = x[:, :, 1::2]
            x, y = x.to(device), y.long().to(device)
            if x.size()[0] != batch_size:
                break
            val_num += x.size(0)
            if block == "phased_LSTM":
                x_decoded, latent, output = model(x, times)
            else:
                x_decoded, latent, output_texture = model(x)

            # construct loss function
            stiff_y, rough_y = property_label(y)
            stiff_y, rough_y = stiff_y.to(device), rough_y.to(device)
            cl_loss = cl_loss_fn(output_texture, y)
            cl_loss_stiffness = recon_loss_fn(latent[:, 0], stiff_y)
            cl_loss_roughness = recon_loss_fn(latent[:, 1], rough_y)
            # compute classification acc
            correct_stiffness += torch.sum((torch.abs(latent[:, 0] - stiff_y) < p_threshold))
            correct_roughness += torch.sum((torch.abs(latent[:, 1] - rough_y) < p_threshold))
            recon_loss = recon_loss_fn(x_decoded, x)
            loss = w_c * cl_loss + w_r * recon_loss
            if use_stiffness:
                loss += w_p * cl_loss_stiffness
            if use_roughness:
                loss +=  w_p * cl_loss_roughness
            
            # compute classification acc
            pred_texture = output_texture.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct_texture += pred_texture.eq(y.data.view_as(pred_texture)).long().cpu().sum().item()
                
            # accumulator
            val_loss += loss.item()
            val_stiffness_loss += cl_loss_stiffness.item()
            val_roughness_loss += cl_loss_roughness.item()
        
        # fill stats
        val_accuracy = correct_texture / val_num
        val_loss /= val_num
        epoch_val_loss.append(val_loss)  # only save the last batch
        epoch_val_acc.append(val_accuracy)
        val_stiffness_accuracy = correct_stiffness.item() / val_num 
        val_stiffness_loss /= val_num        
        epoch_val_stiffness_loss.append(val_stiffness_loss) 
        epoch_val_stiffness_acc.append(val_stiffness_accuracy) 
        val_roughness_accuracy = correct_roughness.item() / val_num 
        val_roughness_loss /= val_num
        epoch_val_roughness_loss.append(val_roughness_loss) 
        epoch_val_roughness_acc.append(val_roughness_accuracy)       
        
        # if epoch < 20 or epoch%200 == 0:
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

    print('Best model at epoch {} with acc {:.3f}'.format(max_val_epoch, max_val_acc))
    training_end =  datetime.now()
    training_time = training_end -training_start 
    print("training takes time {}".format(training_time))

    model.is_fitted = True
    model.eval()

    # TEST at last epoch
    correct_texture = 0
    test_num = 0
    correct_stiffness = 0
    correct_roughness = 0
    
    for i, (XI,  y) in enumerate(test_loader):
        if model.header == 'CNN':
            x = XI
        else:
            x = XB
        # x = x[:, :, 1::2]
        x, y = x.to(device), y.long().to(device)
        
        if x.size(0) != batch_size:
            print(" test batch {} size {} < {}, skip".format(i, x.size()[0], batch_size))
            break
        test_num += x.size(0)
        if block == "phased_LSTM":
            x_decoded, latent, output = model(x, times)
        else:
            x_decoded, latent, output_texture = model(x)

        # compute classification acc
        stiff_y, rough_y = property_label(y)
        stiff_y, rough_y = stiff_y.to(device), rough_y.to(device)
        pred_texture = output_texture.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_texture += pred_texture.eq(y.data.view_as(pred_texture)).long().cpu().sum().item()
        correct_stiffness += torch.sum((torch.abs(latent[:, 0] - stiff_y) < p_threshold))
        correct_roughness += torch.sum((torch.abs(latent[:, 1] - rough_y) < p_threshold))
                
    test_acc1 = correct_texture / test_num #len(test_loader.dataset)
    test_stiffness_acc1 = correct_stiffness.item() / test_num
    test_roughness_acc1 = correct_roughness.item() / test_num
    print('last epoch Test accuracy for', str(kfold_number), ' fold : ', test_acc1, test_stiffness_acc1, test_roughness_acc1)

    # TEST at the best model
    correct_texture = 0
    test_num = 0
    correct_stiffness = 0
    correct_roughness = 0
    saved_dicts = torch.load('models_and_stats/'+model_name+str(max_val_epoch)+'.pt', map_location='cpu')
    model.load_state_dict(saved_dicts['model_state_dict'])

    for i, (XI,  y) in enumerate(test_loader):
        if model.header == 'CNN':
            x = XI
        else:
            x = XB
        # x = x[:, :, 1::2]
        x, y = x.to(device), y.long().to(device)
        
        if x.size(0) != batch_size:
            print(" test batch {} size {} < {}, skip".format(i, x.size()[0], batch_size))
            break
        test_num += x.size(0)
        if block == "phased_LSTM":
            x_decoded, latent, output = model(x, times)
        else:
            x_decoded, latent, output_texture = model(x)

        # compute classification acc
        stiff_y, rough_y = property_label(y)
        stiff_y, rough_y = stiff_y.to(device), rough_y.to(device)
        pred_texture = output_texture.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_texture += pred_texture.eq(y.data.view_as(pred_texture)).long().cpu().sum().item()
        correct_stiffness += torch.sum((torch.abs(latent[:, 0] - stiff_y) < p_threshold))
        correct_roughness += torch.sum((torch.abs(latent[:, 1] - rough_y) < p_threshold))
                
    test_acc2 = correct_texture / test_num #len(test_loader.dataset)
    test_stiffness_acc2 = correct_stiffness.item() / test_num
    test_roughness_acc2 = correct_roughness.item() / test_num
    print('at the best model Test accuracy for', str(kfold_number), ' fold : ', test_acc2, test_stiffness_acc2, test_roughness_acc2)

    # Save stats
    results_dict = {"epoch_train_loss": epoch_train_loss,
                    "epoch_train_acc": epoch_train_acc,
                    "epoch_val_loss": epoch_val_loss,
                    "epoch_val_acc": epoch_val_acc,
                    "test_acc1": test_acc1,
                    "test_acc2": test_acc2,
                    "epoch_train_stiffness_loss": epoch_train_stiffness_loss,
                    "epoch_train_stiffness_acc": epoch_train_stiffness_acc,
                    "epoch_val_stiffness_loss": epoch_val_stiffness_loss,
                    "epoch_val_stiffness_acc": epoch_val_stiffness_acc,
                    "test_stiffness_acc1": test_stiffness_acc1,
                    "test_stiffness_acc2": test_stiffness_acc2,
                    "epoch_train_roughness_loss": epoch_train_roughness_loss,
                    "epoch_train_roughness_acc": epoch_train_roughness_acc,
                    "epoch_val_roughness_loss": epoch_val_roughness_loss,
                    "epoch_val_roughness_acc": epoch_val_roughness_acc,
                    "test_roughness_acc1": test_roughness_acc1,
                    "test_roughness_acc2": test_roughness_acc2}

    dict_name = model_name + '_stats_fold{}_{}.pkl'.format(str(kfold_number), args.rep)
    pickle.dump(results_dict, open(logDir + dict_name, 'wb'))
    print("dump results dict to {}".format(dict_name))

    # assert n_epochs == len(epoch_train_acc), "different epoch length {} {}".format(n_epochs, len(epoch_train_acc))
    # fig, ax = plt.subplots(figsize=(15, 7))
    # ax.plot(np.arange(n_epochs), epoch_train_acc, label="train acc")
    # ax.set_xlabel('epoch')
    # ax.set_ylabel('acc')
    # ax.grid(True)
    # plt.legend(loc='upper right')
    # figname = logDir + model_name +"_train_acc.png"
    # if if_plot:
    #     plt.show()

    plot_stats(logDir + dict_name, model_name)
    return test_acc1, test_stiffness_acc1, test_roughness_acc1

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument("--rand_seed", type=int, default=1, help='random seed')
parser.add_argument("-i", "--rep", type=int, default=0, help='index of running repetition')
parser.add_argument('--data_dir', type=str, default='data', help="DIR set in 'gh_download.sh' to store compiled_data")
parser.add_argument("-k", "--kfold", type=int, default=0, help="kfold_number for loading data")
parser.add_argument("-r", "--reduction", type=int, default=1, help="data reduction ratio for partial training")
parser.add_argument("-c", "--cuda", default=0, help="index of cuda gpu to use")
parser.add_argument("--w_r", default=0.01, type=float, help="weight of recon loss")
parser.add_argument("--w_p", default=5.0, type=float, help="weight of property classification loss")
parser.add_argument("--h_s", default=90, type=int, help="hidden size of rnn layers")
parser.add_argument("--dataset", default='c20', type=str, help="name of dataset")
parser.add_argument("--use_stiffness", default=1, type=int, help="if to use stiffness")
parser.add_argument("--use_roughness", default=1, type=int, help="if to use roughness")
parser.add_argument("--latent_length", default=40, type=int, help="size for the texture latent space")
parser.add_argument("--sequence_length", default=400, type=int, help="length for the input time sequence")
parser.add_argument("--p_threshold", default=0.2, type=float, help="error threshold for property classification")
parser.add_argument("--sample_percent", default=0.6, type=float, help="percentage of samples for training")
parser.add_argument("--i", default=0, type=int, help="seed for splitting training data")


args = parser.parse_args()

# dummy class to replace argparser, if running jupyter notebook
# class Args:
#     rep = 0
#     data_dir = 'data'
#     kfold = 0
#     cuda = '0'
#     reduction = 1
#     r_on = 0.1
#     p_max = 200
#     w_r = 0.01
#     w_p = 1.0
#     h_s = 90
#     dataset = 'c20'
#     stiffness_lat = 10
#     roughness_lat = 10

# args=Args()

args.rand_seed = 66 

result = []
args.sequence_length = 75
args.dataset = 'c20icub'
args.use_stiffness = 0
args.use_roughness = 0
args.w_p = 0

for p in [0.3, 0.2]:
    args.sample_percent = p
    if p == 0.3:
        i_ = [0,1]
    elif p == 0.2:
        i_ = [0,1,2]
    for i in i_: 
        args.i = i
        for k in range(5):
            args.kfold = k
            print(args)
            test_acc1, test_stiffness_acc1, test_roughness_acc1 = train(args)
            result.append([test_acc1, test_stiffness_acc1, test_roughness_acc1])
            print(result)
print(result)                       
