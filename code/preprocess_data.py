#!/usr/bin/env python
# coding: utf-8

'''
Change Log:
16 Oct 2020: add prefix option for getData, getDataset, getLoader, allowing transformation/preprocessing methods like z-normalization. 
    Default: "" (i.e. Nothing) compatible with previous implementation
    "_znorm_": use z normalization for data preprocessing
'''

# In[1]:


import os, sys
import numpy as np
import glob
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split
import random


# In[2]:


num_sample = 50
num_row = 800
seq_len = 400 # (take from the middle)


# In[3]:


# generate pt data from cvs files right after parse_bag.sh in iiwa_stack_ws
# both MatName and SampleNum start from index 1, change to starting from 0
# run twice to ensure every csv is okay
def check_bio_csv_row_number(num_class, num_sample, data_dir, num_row):
    MatName = []
    for i in range(1, num_class+1):
        MatName.append("mat"+str(i))
    for mat in MatName:
        for sample in range(1, num_sample+1):
            bio_csv = data_dir + mat + "/" + mat + "_" + str(sample) +"_bio.csv"
            df = pd.read_csv(bio_csv, header=0, index_col=False)                
            # the header automatically follow when write to csv
            if df.shape[0] == (num_row-1):
                print("append one row to mat {}, sample {}".format(mat, sample))
                last_row = df.iloc[-1:]
                df = df.append(last_row)
                df.to_csv(bio_csv, index=False)
            elif df.shape[0] == (num_row+1):
                print("drop the last row to mat {}, sample {}".format(mat, sample))
                df = df.drop([df.index[-1]])
                df.to_csv(bio_csv, index=False)
            elif df.shape[0] != num_row:
                print(df.shape) # [800, 44]
                print("wierd shape at mat {}, sample {}".format(mat, sample))
            else:
                # row number is just nic
                pass


# In[4]:


def filter_ele_and_timeframe(num_class, num_sample, data_dir, num_row, dict_name="sample_dict.pkl", start=0, end=800):
    print("filter timestamp from {} to {}, seq_len = {}".format(start, end, end-start))
    MatName = []
    for i in range(1, num_class+1):
        MatName.append("mat"+str(i))
    data_dict = {}
    for mat in MatName:
        key = str(int(mat[3:])-1) # label start index from 0
        data_dict[key]={"x":[], "y":[]}
        for sample in range(1, num_sample+1):
            bio_csv = data_dir + mat + "/" + mat + "_" + str(sample) +"_bio.csv"
            df = pd.read_csv(bio_csv, header=0, index_col=False)                
            df = df.drop(labels=['timestamp'], axis=1)
#             print(df.shape) # (800, 44)
            
            # filter ele
            ele_cols = []
            for col in df.columns:
                if 'ele' in col:
                    ele_cols.append(col)
            df_ele = df.loc[:,ele_cols]
#             print(df_ele.shape) # (800, 19)

            df_ele_cut = df_ele.iloc[start:end]
#             print(df_ele_cut.shape) # (400, 19)
            
            # convert to numpy array and transpose
            arr = df_ele_cut.to_numpy().transpose()
#             print(arr.shape) # (19, 400)

            data_dict[key]["x"].append(arr)
            data_dict[key]["y"].append(int(key))
    
    print(data_dict.keys())
#     print(data_dict['0']["x"].shape, data_dict['0']["y"].shape) #(50, 19, 400), (50,)
    print(data_dict['1']["y"])
                
    with open(data_dir+dict_name, "wb") as fout:
        pickle.dump(data_dict, fout)


# ### create torch dataset

# In[5]:


def create_torch_dataset(data_dir, dict_name, dataset_dir, k=0):
    # load all data from data_dict
    with open(data_dir+dict_name, "rb") as fin:
        data_dict = pickle.load(fin)
    X = []
    Y = []
    for key in data_dict:
        X.extend(data_dict[key]["x"])
        Y.extend(data_dict[key]["y"])
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape) # (2500, 19, 400) (2500,)
    
    # perform random split, train:val:test = 6:2:2
    np.random.seed(k)
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.25, shuffle=True, stratify=Y_trainval)
    print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
    
    # check if the dataset is balanced
    unique_elements, counts_elements = np.unique(Y_test, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)))
    
    # save the data in .npy files for fast retrieval
    np.save(dataset_dir+"bio_all.npy", X)
    np.save(dataset_dir+"bio_train_" + str(k) + ".npy", X_train)
    np.save(dataset_dir+"bio_val_" + str(k) + ".npy", X_val)
    np.save(dataset_dir+"bio_test_" + str(k) + ".npy", X_test)
    np.save(dataset_dir+"label_all.npy", Y)
    np.save(dataset_dir+"label_train_" + str(k) + ".npy", Y_train)
    np.save(dataset_dir+"label_val_" + str(k) + ".npy", Y_val)
    np.save(dataset_dir+"label_test_" + str(k) + ".npy", Y_test)


# In[6]:
# add on 19 Oct to chop the seq len
def chop_data_seq_len(X_train, X_val, X_test, Y_train, Y_val, Y_test, seq_len, num_seg=None):
    current_len = X_train.shape[-1]
    assert X_val.shape[-1] == current_len and X_test.shape[-1] == current_len, "different length for train, val, test data"
    if num_seg == None:
        num_seg = current_len // seq_len
    print("chop org data of length {} into {} segments, each of which is has length {}".format(current_len, num_seg, seq_len))
    X_train_seg = []
    X_val_seg = []
    X_test_seg = []
    Y_train_seg = []
    Y_val_seg = []
    Y_test_seg = []
    for i in range(num_seg):
        # print(np.shape(X_train))
        if X_train.shape[-1] == 75:
            X_train_seg.append(X_train[:,:,:,i*seq_len:(i+1)*seq_len])
            X_val_seg.append(X_val[:,:,:,i*seq_len:(i+1)*seq_len])
            X_test_seg.append(X_test[:,:,:,i*seq_len:(i+1)*seq_len])
        else:
            X_train_seg.append(X_train[:,:,i*seq_len:(i+1)*seq_len])
            X_val_seg.append(X_val[:,:,i*seq_len:(i+1)*seq_len])
            X_test_seg.append(X_test[:,:,i*seq_len:(i+1)*seq_len])
        Y_train_seg.append(Y_train)
        Y_val_seg.append(Y_val)
        Y_test_seg.append(Y_test)
        print(len(X_train_seg))
        print(X_train.shape)
        if X_train.shape[-1] == 75:
            num_seg, num_sample, num_feature1, num_feature2, seq_len = np.array(X_train_seg).shape
            X_train = np.array(X_train_seg).reshape((-1, num_feature1, num_feature2, seq_len))
            X_val = np.array(X_val_seg).reshape((-1, num_feature1, num_feature2, seq_len))
            X_test = np.array(X_test_seg).reshape((-1, num_feature1, num_feature2, seq_len))
        else:
            num_seg, num_sample, num_feature, seq_len = np.array(X_train_seg).shape
            X_train = np.array(X_train_seg).reshape((-1, num_feature, seq_len))
            X_val = np.array(X_val_seg).reshape((-1, num_feature, seq_len))
            X_test = np.array(X_test_seg).reshape((-1, num_feature, seq_len))
            Y_train = np.array(Y_train_seg).reshape((-1))
            Y_val = np.array(Y_val_seg).reshape((-1))
            Y_test = np.array(Y_test_seg).reshape((-1))
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
        
        


def get_TrainValTestData(path, k=0, prefix="", seq_len=400, num_seg=1, p=None, i=None):
    X_train = np.load(path + 'bio_train_' + prefix + str(k) + '.npy')
    X_val = np.load(path + 'bio_val_' + prefix + str(k) + '.npy')
    # X_test = np.load(path + 'bio_test_' + prefix + str(k) + '.npy')
    try:
        X_test = np.load(path + 'bio_test_' + prefix + str(k) + '.npy')
        Y_test = np.load(path + 'label_test_' + prefix + str(k) + '.npy')
    except FileNotFoundError:
        X_test = np.load(path + 'bio_test' + prefix + '.npy')
        Y_test = np.load(path + 'labels_test' + prefix + '.npy')
    try:
        Y_train = np.load(path + 'label_train_' + str(k) + '.npy')
    except FileNotFoundError:
        Y_train = np.load(path + 'labels_train_' + str(k) + '.npy')
    try:
        Y_val = np.load(path + 'label_val_' + str(k) + '.npy')
    except FileNotFoundError:
        Y_val = np.load(path + 'labels_val_' + str(k) + '.npy')
        # Y_test = np.load(path + 'label_test_' + str(k) + '.npy')
    if p is not None:
        np.random.seed(6)
        perm_idx = np.random.permutation(len(Y_train))
        X_train = X_train[perm_idx]
        Y_train = Y_train[perm_idx]
        n_X_train = X_train.shape[0]
    if p == 0.3:
        print(int(i*n_X_train/2), ':', int(i*n_X_train/2+n_X_train/2))
        X_train = X_train[int(i*n_X_train/2):int(i*n_X_train/2+n_X_train/2), :, :]
        Y_train = Y_train[int(i*n_X_train/2):int(i*n_X_train/2+n_X_train/2)]
        print('X_train', X_train.shape)
    if p == 0.2:
        print(int(i*n_X_train/3),':',int(i*n_X_train/3+n_X_train/3))
        X_train = X_train[int(i*n_X_train/3):int(i*n_X_train/3+n_X_train/3), :, :]
        Y_train = Y_train[int(i*n_X_train/3):int(i*n_X_train/3+n_X_train/3)]
        print('X_train', X_train.shape)

    X_train, X_val, X_test, Y_train, Y_val, Y_test = chop_data_seq_len(X_train, X_val, X_test, Y_train, Y_val, Y_test, seq_len, num_seg=num_seg)
    return torch.FloatTensor(X_train), torch.FloatTensor(X_val), torch.FloatTensor(X_test), torch.FloatTensor(Y_train),torch.FloatTensor(Y_val), torch.FloatTensor(Y_test)

# In[7]:


def get_TrainValTestLoader(path, k=0, batch_size=8, shuffle=True, prefix="", seq_len=400, p=None, i=None):
    X_train, X_val, X_test, Y_train, Y_val, Y_test = get_TrainValTestData(path, k, prefix=prefix, seq_len=seq_len, p=p, i=i)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=shuffle,batch_size=batch_size)

    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset,shuffle=shuffle,batch_size=batch_size)
    
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=shuffle,batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


# In[ ]:


def get_TrainValTestDataset(path, k=0, batch_size=8, shuffle=True, prefix="", seq_len=400, p=None):
    X_train, X_val, X_test, Y_train, Y_val, Y_test = get_TrainValTestData(path, k, prefix=prefix, seq_len=seq_len)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    return train_dataset, val_dataset, test_dataset


# In[8]:


def create_loaders_50_50():
    num_class = 50
    data_dir = "/home/ruihan/Documents/mat_data_2020/"
    check_bio_csv_row_number(num_class, num_sample, data_dir, num_row)
    dict_name = "class_dict_50_50.pkl"     
    filter_ele_and_timeframe(num_class, num_sample, data_dir, num_row, dict_name, start=200, end=600)
    dataset_dir = "/home/ruihan/Desktop/HCNC/data/c50/"
    create_torch_dataset(data_dir, dict_name, dataset_dir, k=0)
    train_loader, val_loader, test_loader = get_TrainValTestLoader(dataset_dir, k=0, batch_size=8, shuffle=True)


# In[9]:


# create train, val, test loader for the first 20 classes with new data
def create_loaders_20_50_new():
    num_class = 20
    data_dir = "/home/ruihan/Documents/mat_data_2020/"
    check_bio_csv_row_number(num_class, num_sample, data_dir, num_row)
    dict_name = "class_dict_20_50_new.pkl"  
    filter_ele_and_timeframe(num_class, num_sample, data_dir, num_row, dict_name, start=200, end=600)
    dataset_dir = "/home/ruihan/Desktop/HCNC/data/c20new/"
    create_torch_dataset(data_dir, dict_name, dataset_dir, k=0)
#     train_loader, val_loader, test_loader = get_TrainValTestLoader(dataset_dir, k=0, batch_size=8, shuffle=True)


# In[10]:


def create_loaders_20_50():
    num_class = 20
    data_dir = "/home/ruihan/Documents/material_data_2020Mar/"
    check_bio_csv_row_number(num_class, num_sample, data_dir, num_row)
    dict_name = "class_dict_20_50.pkl"  
    filter_ele_and_timeframe(num_class, num_sample, data_dir, num_row, dict_name, start=200, end=600)
    dataset_dir = "/home/ruihan/Desktop/HCNC/data/c20/"
    create_torch_dataset(data_dir, dict_name, dataset_dir, k=0)
#     train_loader, val_loader, test_loader = get_TrainValTestLoader(dataset_dir, k=0, batch_size=8, shuffle=True)


# In[11]:


# create_loaders_20_50()


# In[12]:


# create_loaders_20_50_new()


# In[13]:


# create_loaders_50_50()


def property_label(y, const=1):
    stiff_y = []
    rough_y = []

    for i in range(len(y)):
        if y[i] in [2, 10, 17]: 
            stiff_y.append(-1*const)  # 0
        elif y[i] in [1, 3, 4, 5, 6, 12]:
            stiff_y.append(0) # 0.5
        else:
            stiff_y.append(1*const) # 1

        if y[i] in [3, 4, 7, 9, 12, 13, 14, 19]: 
            rough_y.append(-1*const)
        elif y[i] in [5, 8, 11, 15, 16, 17]:
            rough_y.append(0)
        else:
            rough_y.append(1*const)

    stiff_y, rough_y = torch.FloatTensor(stiff_y), torch.FloatTensor(rough_y)

    return stiff_y, rough_y

def c50_property_label(y):
    stiff_y = []
    rough_y = []

    for i in range(len(y)):
        if y[i] in [2, 10, 17, 29]: 
            stiff_y.append(-1)  # 0
        elif y[i] in [1, 3, 4, 5, 6, 12, 20, 21, 25]:
            stiff_y.append(0) # 0.5
        else:
            stiff_y.append(1) # 1

        if y[i] in [3, 4, 7, 9, 12, 13, 14, 19, 21, 22, 26, 38, 43, 49]: 
            rough_y.append(-1)
        elif y[i] in [5, 8, 11, 15, 16, 17, 20, 24, 27, 30, 31, 32, 34, 35, 37, 41, 44, 46, 48]:
            rough_y.append(0)
        else:
            rough_y.append(1)
            
    stiff_y, rough_y = torch.FloatTensor(stiff_y), torch.FloatTensor(rough_y)
    return stiff_y, rough_y

def process_c50(path='data/c50/', dataset_dir='data/c30/'):
    # change y label
    # add property labels
    x = np.load(path + 'bio_all.npy')
    y = np.load(path + 'label_all.npy')
    over_20_0_1 = y >= 20
    over_20_index = np.asarray(np.nonzero(over_20_0_1))
    # print(over_20_index)
    # print(over_20_index.shape)
    c30_x = np.squeeze(x[over_20_index])
    c30_y = np.squeeze(y[over_20_index])
    print(c30_x.shape, c30_y.shape)
    # print(c30_x, c30_y)
    np.save(dataset_dir+"bio_all.npy", c30_x)
    np.save(dataset_dir+"label_all.npy", c30_y)

def c30_process_data(c30_x, c30_y):
    c10_idx = []
    c10_y = []
    for n, i in enumerate(c30_y):
        if i in [20, 21, 22, 23, 25, 28, 29, 33, 34, 40]:
            c10_idx.append(n)
            c10_y.append(i)
    c10_idx = torch.tensor(c10_idx)
    c10_x = torch.index_select(c30_x, 0, c10_idx)
    c10_stiff_y, c10_rough_y = c50_property_label(c30_y[c10_idx])
    return c10_x, c10_y, c10_stiff_y, c10_rough_y
# process_c50()

# original
#     if y[i] in [2, 3, 5, 6, 9, 10, 12, 17]: 
#             stiff_y.append(0)
#         else:
#             stiff_y.append(1)

#         if y[i] in [3, 4, 5, 7, 8, 9, 12, 13, 14, 19]: 
#             rough_y.append(0)
#         else:
#             rough_y.append(1)
    
